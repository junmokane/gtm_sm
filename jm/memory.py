import numpy as np
from sklearn.metrics import pairwise_distances


class Spatial_Memory_FIFO():
    def __init__(self, memory_size, batch_size, s_dim):
        super(Spatial_Memory_FIFO, self).__init__()
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = np.zeros((batch_size, memory_size, s_dim), np.float32)
        self.memory_time_index = np.zeros((batch_size, memory_size), np.int32)
        self.cur_size = 0
        self.cur_t = 0
        self.fifo_index = 0
    
    def write(self, st_observation_t):
        '''
        st_observation_t : np array (batch_size, s_dim)
        '''
        self.memory[:, self.fifo_index, :] = st_observation_t
        self.memory_time_index[:, self.fifo_index] = self.cur_t
        
        self.fifo_index = (self.fifo_index + 1) % self.memory_size
        if self.cur_size < self.memory_size:
            self.cur_size += 1
        self.cur_t += 1
    
    def get_memory_time_index(self):
        return self.memory_time_index[:, :self.cur_size]
    
    def get_memory(self):
        return self.memory[:, :self.cur_size, :]


class Spatial_Memory_Heuristics_V1():
    '''
    This memory writes the new sample in the slot which 
    has the minimum distance over every two slots
    '''
    def __init__(self, memory_size, batch_size, s_dim):
        super(Spatial_Memory_Heuristics_V1, self).__init__()
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = np.zeros((batch_size, memory_size, s_dim), np.float32)
        self.memory_time_index = np.zeros((batch_size, memory_size), np.int32)
        self.cur_size = 0
        self.cur_t = 0
    
    def write(self, st_observation_t):
        '''
        st_observation_t : np array (batch_size, s_dim)
        '''
        if self.cur_size < self.memory_size:
            self.memory[:, self.cur_size, :] = st_observation_t
            self.memory_time_index[:, self.cur_size] = self.cur_t
        else:
            for index_sample in range(self.batch_size):
                pairwise_dist = pairwise_distances(self.memory[index_sample])  # [memory_size, memory_size]
                sort_dist = np.sort(pairwise_dist, axis=1)  # [memory_size, memory_size]
                dist = np.sum(sort_dist[:, :3], axis=1)  # [memory_size,]
                min_dist_index = np.argmin(dist)
                
                # print(pairwise_dist[5])
                # print(sort_dist[5])
                # print(dist)
                # print(min_dist_index)
                # exit()
                
                self.memory[index_sample, min_dist_index] = st_observation_t[index_sample]
                self.memory_time_index[index_sample, min_dist_index] = self.cur_t
        
        if self.cur_size < self.memory_size:
            self.cur_size += 1
        self.cur_t += 1
        
        '''
        # Heuristic Oracle: save globally uniform time step. Only works for explore_walk_wo_wall  
        if self.cur_size < self.memory_size and self.cur_t in [72, 75, 77, 80, 69, 6, 8, 51, 67, 4, 2, 53, 64, 61, 59, 56]:
            self.memory[:, self.cur_size, :] = st_observation_t
            self.memory_time_index[:, self.cur_size] = self.cur_t
            self.cur_size += 1
        self.cur_t += 1
        '''
                
    def get_memory_time_index(self):
        return self.memory_time_index[:, :self.cur_size]
    
    def get_memory(self):
        return self.memory[:, :self.cur_size, :]