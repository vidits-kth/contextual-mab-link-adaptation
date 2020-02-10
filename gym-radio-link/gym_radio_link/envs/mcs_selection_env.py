from __future__ import absolute_import

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import itpp
import time

from .src import *

class MCSSelectionEnv(gym.Env):
    ''' Define environment parameters that are common to all instances
        of the environment
    '''
    dirpath = os.path.dirname(os.path.abspath(__file__))
    offline_model_datafile = dirpath + '/src/Offline_Link_Model.npy'
    offline_model = np.load( offline_model_datafile, encoding='latin1', allow_pickle = True )[ ( ) ]

    snr_vs_bler = np.array(offline_model['snr_vs_bler'])
    snr_range_dB = np.array(offline_model['snr_range_dB'])

    #ignore_indices = [0, 9, 15, 19, 20, 21]
    #snr_vs_bler = np.delete(snr_vs_bler, ignore_indices, axis=1)
    #snr_range_dB

    nrof_snr, nrof_mcs = snr_vs_bler.shape
    nrof_resource_blocks = 1

    mcs = {'modulation_order': offline_model['modulation_order'],
           'transport_block_size': offline_model['block_size']}

#    nrof_mcs = 22
#    nrof_resource_blocks = 1

    # Load AWGN data used to calculate the CQI
#    dirpath = os.path.dirname(os.path.abspath(__file__))
#    awgn_datafile = dirpath + '/src/CONSTANTS/AWGN_CUSTOM_CONFIG_DATAFILE.npy'
#    awgn_data = load_from_file(awgn_datafile, encoding='latin1')

#    mcs = {'modulation_order': [],
#          'transport_block_size': []}
#    
#    for i in range(nrof_mcs):
#        modorder, tbs = get_transmission_parameters_from_cqi(i, nrof_resource_blocks)
#        mcs['modulation_order'].append(modorder)
#        mcs['transport_block_size'].append(tbs)
 
    ''' Construct an instance of the environment. The parameters
        ue_relative_speed: Determines the Doppler
        interference_prob: Probability of random Gaussian interefence in a TTI
        cqi_reporting_interval: Periodicity of the Channel Quality Index (CQI)
    '''
    def __init__(self, 
                 ue_relative_speed=0.83, 
                 intercell_interference=False,
                 cqi_reporting_interval=1):

        # Set the member variables (fixed over the simulation)
        self.intercell_interference = intercell_interference
        self.ue_relative_speed = ue_relative_speed # m/s 

        self.cqi_reporting_interval = cqi_reporting_interval

        self.bler_target = 0.1           
        self.channel_spec = itpp.comm.Channel_Specification(itpp.comm.CHANNEL_PROFILE.ITU_Vehicular_B)

        self.seed_value = None           # Initialized in self.seed(...)

        self.ue_noise_variance = None    # Initialized in self.reset()
        self.channel_coefficients = None # Initialized in self.reset()

        # State variables that are updated in every TTI
        self.subframe_index = None
        self.cqi = None                  # Initialized in self.reset()

        # Setup the radio channel
        self.channel = None              # Initialized in self.reset()

    '''Random number generator seed for repeatability
    '''
    def seed(self, seed_value=42):
        self.seed_value = seed_value

        # Set the random number generator seed
        itpp.RNG_reset(self.seed_value)

        if int(self.ue_relative_speed) == -1:
#            ue_speed_kmph = itpp.random.randi(1, 120)
            ue_speed_kmph = max(5.0, 30.0 + 10.0 * itpp.random.randn() )
            self.ue_relative_speed = ue_speed_kmph * 1000.0 / 3600.0 # m/s 
        else:
            ue_speed_kmph = int(self.ue_relative_speed * 3600 / 1000)

        # Initialize the random variables
        snr_dB = 20.0 + 5.0 * itpp.random.randn()
#        snr_dB = itpp.random.randi(5, 25)
        #self.ue_noise_variance = itpp.math.inv_dB(-1.0 * float(snr_dB))
        self.ue_noise_variance = 10 ** ( -0.1 * float(snr_dB) )  # Inverse dB

        # Setup the channel
        self.channel = setup_fading_channel(self.channel_spec, self.ue_relative_speed)

        return (snr_dB, ue_speed_kmph)
        
    ''' Simulate the next TTI
        The modulation and coding scheme (MCS) is provided as input
    '''
    def step(self, mcs_index):
        # Simulate random intercell interference 
        interf_flag = 0
        sir_dB = 10 
        if self.intercell_interference:
            #if ((int(self.subframe_index / 15)) % 3) == 0:
            if itpp.randu() < 0.2:
                #interf_variance = itpp.math.inv_dB(-1.0 * float(sir_dB))
                interf_variance = 10 ** (-0.1 * float(sir_dB))  # Inverse dB
                interf_flag = 1
            else:
                interf_variance = 0.0
                interf_flag = 0

        else:
            interf_variance = 0.0
    
        # Determine the transmission parameters based on the MCS 
        modulation_order = self.mcs['modulation_order'][mcs_index]
        transport_block_size = self.mcs['transport_block_size'][mcs_index]
    
        # If the number of scheduled transport bits is zero, skip transmitting in this subframe
        if transport_block_size == 0:
            #print ('Subframe %d: UE is out of range, skipping transmission!'%(self.subframe_index))

            harq_ack = False
            subframe_throughput = 0
        else:
            # Generate new transport bits and update buffers
            transport_bits = itpp.randb(transport_block_size)
        
            transmit_buffer, interleaver_sequence = channel_encode_and_interleave_bits(transport_bits)
            receive_buffer = itpp.zeros(transmit_buffer.length())
        
            # Extract the bits to be transmitted from the buffer
            nrof_transmit_bits = calculate_nrof_transmit_bits(modulation_order, self.nrof_resource_blocks)
            transmit_bits = extract_next_bits_with_wraparound(transmit_buffer, 0, nrof_transmit_bits)
        
            # Propagate the transmit bits over the channel and update the receive HARQ buffer
            received_soft_values = propagate_transmit_bits_over_channel(transmit_bits, modulation_order, self.nrof_resource_blocks, self.channel_coefficients, self.ue_noise_variance + interf_variance)
            receive_buffer = add_values_with_wraparound(receive_buffer, 0, received_soft_values)
        
            # Extract the transport bits from the receive HARQ buffer 
            decoded_bits = deinterleave_and_channel_decode_symbols(receive_buffer, interleaver_sequence)
        
            # Determine result of the transmission
            if (decoded_bits == transport_bits):
                subframe_throughput = transport_block_size
                harq_ack = True
            else:
                subframe_throughput = 0
                harq_ack = False
                
        # snr_per_subcarrier = itpp.math.pow(itpp.math.abs(self.channel_coefficients), 2) * (1.0 / self.ue_noise_variance)
        
        # Print transmission information
        #print('Subframe %d, CQI: %d, MCS Index: %d, BLER: %0.4f, TBS: %d, HARQ ACK: %d'%(self.subframe_index, self.cqi, mcs_index, self.bler_target, transport_block_size, harq_ack))

        self.subframe_index += 1

        # Obtain the channel coefficients (frequency-domain complex channel coefficients).   
        self.channel_coefficients = calculate_channel_frequency_response(self.channel, self.subframe_index)
    
        #  Obtain the wideband channel quality index (CQI) at reporting intervals
        if (self.subframe_index % self.cqi_reporting_interval) == 0:
            self.cqi = calculate_wideband_channel_quality_index(self.channel_coefficients, self.ue_noise_variance + interf_variance, self.bler_target, self.offline_model)

        return (self.subframe_index, subframe_throughput, self.cqi, harq_ack, interf_flag) #, snr_per_subcarrier.to_numpy_ndarray())
    
    '''
    ''' 
    def reset(self):
        # Reset the random number generator seed
        itpp.RNG_reset(self.seed_value)

        # Reset the subframe index
        self.subframe_index = 0

        # Obtain the initial channel coefficients (frequency-domain complex channel coefficients).   
        self.channel_coefficients = calculate_channel_frequency_response(self.channel, self.subframe_index)

        self.cqi = calculate_wideband_channel_quality_index(self.channel_coefficients, self.ue_noise_variance, self.bler_target, self.offline_model)

        return self.cqi

    def render(self, mode='random', close=False):
        pass
