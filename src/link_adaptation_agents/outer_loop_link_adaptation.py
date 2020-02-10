import numpy as np

from . import _channel_quality_index as lib

class OuterLoopLinkAdaptation():
    def __init__(self, bler_target, olla_window_size, awgn_data, olla_step_size = 0.1):
        self.agent_type = 'base'

        self.bler_target = bler_target
        self.olla_window_size = olla_window_size

        self.sinr_offset = 0
        self.acks = np.zeros((self.olla_window_size), dtype=np.bool_)

        self.olla_step_size = olla_step_size

        self.awgn_data = awgn_data
        
        _, nrof_cqi = awgn_data['snr_vs_bler'].shape
        
        self.cqi_to_estimated_sinr =  [lib.estimate_sinr_from_cqi(self.awgn_data, i) for i in range(nrof_cqi)]

    def reset( self ):
        self.sinr_offset = 0
        self.acks = np.zeros((self.olla_window_size), dtype=np.bool_)

    def update_agent(self, ack):
        if ack == 1:
            self.sinr_offset -=  self.olla_step_size
        else:
            self.sinr_offset += self.olla_step_size * (1 - self.bler_target) / self.bler_target
      
    def determine_action_indices(self, cqi):
        
        mcs = []

        if cqi == 0:
            mcs = 0
        else:
            estimated_sinr = self.cqi_to_estimated_sinr[cqi]
    
            adjusted_sinr = estimated_sinr - self.sinr_offset
    
            mcs = lib.determine_mcs_from_sinr(self.awgn_data, adjusted_sinr, self.bler_target)

        return mcs

    
    def determine_predicted_success(self, cqi):
        estimated_sinr = self.cqi_to_estimated_sinr[cqi]
    
        adjusted_sinr = estimated_sinr - self.sinr_offset
    
        bler = lib.determine_bler_at_sinr(self.awgn_data, adjusted_sinr)
        
        success_rate = []
        for i in range( bler.length( ) ):
            success_rate.append( 1.0 - bler[ i ] )
        
        return np.array( success_rate )    