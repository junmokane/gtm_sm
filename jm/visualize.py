import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def show_trajectory(position, goto_ran_len_list, model):
    fig = plt.figure()
    fig.clf()
    t = model.observe_dim
    plt.suptitle('Sample Trajectory', fontsize=25)
    gs = gridspec.GridSpec(10, 20)
    sample_id = np.random.randint(0, model.batch_size, size=(1))
    goto_ran_len = goto_ran_len_list[sample_id[0]]
    
    ax1 = plt.subplot(gs[0:5, 0:5])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Global exploration (81)')
    ax1.set_aspect('equal')
    plt.axis([-1, 9, -1, 9])
    plt.gca().invert_yaxis()
    plt.plot(position[sample_id, 1, 0:81].T, position[sample_id, 0, 0:81].T, color='k',
                linestyle='solid', marker='o')
    plt.plot(position[sample_id, 1, 80], position[sample_id, 0, 80], 'bs')
     
    ax2 = plt.subplot(gs[0:5, 7:12])
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f'Local exploration ({goto_ran_len}+{t - 81 - goto_ran_len})')
    ax2.set_aspect('equal')
    plt.axis([-1, 9, -1, 9])
    plt.gca().invert_yaxis()
    plt.plot(position[sample_id, 1, 80: t].T, position[sample_id, 0, 80: t].T, color='k',
                linestyle='solid', marker='o')
    plt.plot(position[sample_id, 1, t - 1], position[sample_id, 0, t - 1], 'bs')
    
    ax3 = plt.subplot(gs[0:5, 14:19])
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title(f'Prediction phase ({model.total_dim - t})')
    ax3.set_aspect('equal')
    plt.axis([-1, 9, -1, 9])
    plt.gca().invert_yaxis()
    plt.plot(position[sample_id, 1, t:].T, position[sample_id, 0, t:].T, color='k',
                linestyle='solid', marker='o')
    plt.plot(position[sample_id, 1, -1], position[sample_id, 0, -1], 'bs')
    
    plt.savefig('./jm/traj.png')
    plt.close()


def show_experiment(model, x, st_observation_list, 
                    st_prediction_list, xt_prediction_list, 
                    position, goto_ran_len_list, memory_time_index):
    '''
    x                           tensor  (self.batch_size, 3, 32, 32)
    st_observation_list         list    (self.observe_dim)(self.batch_size, self.s_dim)
    st_prediction_list          list    (self.total_dim - self.observe_dim)(self.batch_size, self.s_dim)
    xt_prediction_list          list    (self.total_dim - self.observe_dim)(self.batch_size, self.x_dim)
    position                    numpy   (self.batch_size, self.s_dim, self.total_dim)
    goto_ran_len_list           list    (self.batch_size)
    memory_time_index           numpy   (self.batch_size, self.memory_size)
    '''    
    # trajectory initialization
    sample_id = np.random.randint(0, model.batch_size, size=(1))[0]
    goto_ran_len = goto_ran_len_list[sample_id]
    
    st_observation_sample = np.zeros((model.observe_dim, model.s_dim))
    for t in range(model.observe_dim):
        st_observation_sample[t] = st_observation_list[t][sample_id].cpu().detach().numpy()

    st_prediction_sample = np.zeros((model.total_dim - model.observe_dim, model.s_dim))
    for t in range(model.total_dim - model.observe_dim):
        st_prediction_sample[t] = st_prediction_list[t][sample_id].cpu().detach().numpy()

    st_2_max = np.maximum(np.max(st_observation_sample[:, 0]), np.max(st_prediction_sample[:, 0]))
    st_2_min = np.minimum(np.min(st_observation_sample[:, 0]), np.min(st_prediction_sample[:, 0]))
    st_1_max = np.maximum(np.max(st_observation_sample[:, 1]), np.max(st_prediction_sample[:, 1]))
    st_1_min = np.minimum(np.min(st_observation_sample[:, 1]), np.min(st_prediction_sample[:, 1]))
    axis_st_1_max = st_1_max + (st_1_max - st_1_min) / 10.0
    axis_st_1_min = st_1_min - (st_1_max - st_1_min) / 10.0
    axis_st_2_max = st_2_max + (st_2_max - st_2_min) / 10.0
    axis_st_2_min = st_2_min - (st_2_max - st_2_min) / 10.0
    
    obs_dim = model.observe_dim
    
    # plot trajectory of position and inferred state
    '''
    gs = fig.add_gridspec(3, 10)
    f_ax1 = fig.add_subplot(gs[0:2, 0:2])
    f_ax1.set_title('gs[0:2, 0:2]')
    for i in range(8):
        f_ax = fig.add_subplot(gs[0, i+2])
        f_ax.set_title(f'gs[0, {i+2}]')
        f_ax = fig.add_subplot(gs[1, i+2])
        f_ax.set_title(f'gs[1, {i+2}]')
        f_ax = fig.add_subplot(gs[2, i+2])
        f_ax.set_title(f'gs[2, {i+2}]')
    
    plt.savefig('./jm/grid_test.png')
    plt.close()
    '''
    
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    fig.clf()
    gs = fig.add_gridspec(2, 3)
    
    # plot position trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Global exploration (81 steps)')
    ax1.set_aspect('equal')
    plt.axis([-1, 9, -1, 9])
    plt.gca().invert_yaxis()
    plt.plot(position[sample_id, 1, 0:81].T, position[sample_id, 0, 0:81].T, color='k',
             linestyle='solid', marker='o')
    plt.plot(position[sample_id, 1, 80], position[sample_id, 0, 80], 'bs')
     
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f'Local exploration ({goto_ran_len}+{obs_dim - 81 - goto_ran_len} steps)')
    ax2.set_aspect('equal')
    plt.axis([-1, 9, -1, 9])
    plt.gca().invert_yaxis()
    plt.plot(position[sample_id, 1, 80:obs_dim].T, position[sample_id, 0, 80:obs_dim].T, color='k',
             linestyle='solid', marker='o')
    plt.plot(position[sample_id, 1, obs_dim - 1], position[sample_id, 0, obs_dim - 1], 'bs')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title(f'Prediction phase ({model.total_dim - obs_dim} steps)')
    ax3.set_aspect('equal')
    plt.axis([-1, 9, -1, 9])
    plt.gca().invert_yaxis()
    plt.plot(position[sample_id, 1, obs_dim:].T, position[sample_id, 0, obs_dim:].T, color='k',
             linestyle='solid', marker='o')
    plt.plot(position[sample_id, 1, -1], position[sample_id, 0, -1], 'bs')
    
    # plot st trajectory
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_xlabel('$s_1$')
    ax4.set_ylabel('$s_2$')
    ax4.set_title('Inferred states')
    ax4.set_aspect('equal')
    plt.axis([axis_st_1_min, axis_st_1_max, axis_st_2_min, axis_st_2_max])
    plt.gca().invert_yaxis()
    plt.plot(st_observation_sample[0:81, 1].T, st_observation_sample[0:81, 0].T, color='k',
             linestyle='solid', marker='o')
    plt.plot(st_observation_sample[80, 1], st_observation_sample[80, 0], 'bs')
     
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_xlabel('$s_1$')
    ax5.set_ylabel('$s_2$')
    ax5.set_title('Inferred states')
    ax5.set_aspect('equal')
    plt.axis([axis_st_1_min, axis_st_1_max, axis_st_2_min, axis_st_2_max])
    plt.gca().invert_yaxis()
    plt.plot(st_observation_sample[80:obs_dim, 1].T, st_observation_sample[80:obs_dim, 0].T, color='k',
             linestyle='solid', marker='o')
    plt.plot(st_observation_sample[obs_dim - 1, 1], st_observation_sample[obs_dim - 1, 0], 'bs')
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_xlabel('$s_1$')
    ax6.set_ylabel('$s_2$')
    ax6.set_title('Inferred states')
    ax6.set_aspect('equal')
    plt.axis([axis_st_1_min, axis_st_1_max, axis_st_2_min, axis_st_2_max])
    plt.gca().invert_yaxis()
    plt.plot(st_prediction_sample[0:, 1].T, st_prediction_sample[0:, 0].T, color='k',
             linestyle='solid', marker='o')
    plt.plot(st_prediction_sample[-1, 1], st_prediction_sample[-1, 0], 'bs')
    
    plt.savefig('./jm/exp_traj.png')
    plt.close()
    
    # memory evaluation initialization
    selected_time_list = [obs_dim, obs_dim + 1, obs_dim + 2, obs_dim + 32, 
                          obs_dim + 52, obs_dim + 72, obs_dim + 92, model.total_dim - 1]
    
    # plot memory evaluation
    fig = plt.figure(figsize=(30, 10), constrained_layout=True)
    fig.clf()
    gs = fig.add_gridspec(3, 10)
    
    # plot original image 
    sample_imgs = x[sample_id]  # [3,32,32]
    sample_imgs_t = np.copy(sample_imgs.cpu().detach().numpy())
    sample_memory_time_index = memory_time_index[sample_id]
    
    for index_time in sample_memory_time_index:
        position_h_t = np.asscalar(position[sample_id, 0, index_time])
        position_w_t = np.asscalar(position[sample_id, 1, index_time])
        sample_imgs_t[0, 3 * position_h_t, 3 * position_w_t] = 1.0
        sample_imgs_t[1, 3 * position_h_t, 3 * position_w_t] = 0.0
        sample_imgs_t[2, 3 * position_h_t, 3 * position_w_t] = 0.0
        # sample_imgs_t[0, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t] = 1.0
        # sample_imgs_t[0, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t + 8 - 1] = 1.0
        # sample_imgs_t[0, 3 * position_h_t, 3 * position_w_t: 3 * position_w_t + 8] = 1.0
        # sample_imgs_t[0, 3 * position_h_t + 8 - 1, 3 * position_w_t: 3 * position_w_t + 8] = 1.0
        # sample_imgs_t[1, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t] = 0.0
        # sample_imgs_t[1, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t + 8 - 1] = 0.0
        # sample_imgs_t[1, 3 * position_h_t, 3 * position_w_t: 3 * position_w_t + 8] = 0.0
        # sample_imgs_t[1, 3 * position_h_t + 8 - 1, 3 * position_w_t: 3 * position_w_t + 8] = 0.0
        # sample_imgs_t[2, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t] = 0.0
        # sample_imgs_t[2, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t + 8 - 1] = 0.0
        # sample_imgs_t[2, 3 * position_h_t, 3 * position_w_t: 3 * position_w_t + 8] = 0.0
        # sample_imgs_t[2, 3 * position_h_t + 8 - 1, 3 * position_w_t: 3 * position_w_t + 8] = 0.0
    
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.set_title('Original image', fontsize=30)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_aspect('equal')
    plt.axis('off')
    plt.imshow(sample_imgs_t.transpose((1, 2, 0)))
    
    # plot images in time step
    for i, selected_time in enumerate(selected_time_list):
        sample_imgs_t = np.copy(sample_imgs.cpu().detach().numpy())    
        position_h_t = np.asscalar(position[sample_id, 0, selected_time])
        position_w_t = np.asscalar(position[sample_id, 1, selected_time])
        sample_imgs_t[0, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t] = 1.0
        sample_imgs_t[0, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t + 8 - 1] = 1.0
        sample_imgs_t[0, 3 * position_h_t, 3 * position_w_t: 3 * position_w_t + 8] = 1.0
        sample_imgs_t[0, 3 * position_h_t + 8 - 1, 3 * position_w_t: 3 * position_w_t + 8] = 1.0
        sample_imgs_t[1, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t] = 1.0
        sample_imgs_t[1, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t + 8 - 1] = 1.0
        sample_imgs_t[1, 3 * position_h_t, 3 * position_w_t: 3 * position_w_t + 8] = 1.0
        sample_imgs_t[1, 3 * position_h_t + 8 - 1, 3 * position_w_t: 3 * position_w_t + 8] = 1.0
        sample_imgs_t[2, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t] = 0.0
        sample_imgs_t[2, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t + 8 - 1] = 0.0
        sample_imgs_t[2, 3 * position_h_t, 3 * position_w_t: 3 * position_w_t + 8] = 0.0
        sample_imgs_t[2, 3 * position_h_t + 8 - 1, 3 * position_w_t: 3 * position_w_t + 8] = 0.0
        sample_imgs_t[0, 3 * position_h_t, 3 * position_w_t] = 0.0
        sample_imgs_t[1, 3 * position_h_t, 3 * position_w_t] = 1.0
        sample_imgs_t[2, 3 * position_h_t, 3 * position_w_t] = 0.0
    
        for index_time in sample_memory_time_index:
            position_h_t = np.asscalar(position[sample_id, 0, index_time])
            position_w_t = np.asscalar(position[sample_id, 1, index_time])
            sample_imgs_t[0, 3 * position_h_t, 3 * position_w_t] = 1.0
            sample_imgs_t[1, 3 * position_h_t, 3 * position_w_t] = 0.0
            sample_imgs_t[2, 3 * position_h_t, 3 * position_w_t] = 0.0
    
        ax1 = fig.add_subplot(gs[0, i + 2])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        ax1.set_title(f't={selected_time}', fontsize=30)
        plt.axis('off')
        plt.imshow(sample_imgs_t.transpose((1, 2, 0)))
    
    # plot zoomed images in time step
    for i, selected_time in enumerate(selected_time_list):
        sample_imgs_t = np.copy(sample_imgs.cpu().detach().numpy())
        position_h_t = np.asscalar(position[sample_id, 0, selected_time])
        position_w_t = np.asscalar(position[sample_id, 1, selected_time])
        sample_imgs_local_t = sample_imgs_t[:, 3 * position_h_t: 3 * position_h_t + 8, 3 * position_w_t: 3 * position_w_t + 8]
        
        ax1 = fig.add_subplot(gs[1, i + 2])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        ax1.set_title('Zoom-in', fontsize=30)
        plt.axis('off')
        plt.imshow(sample_imgs_local_t.transpose((1, 2, 0)))
    
    # plot inferred images in time step
    for i, selected_time in enumerate(selected_time_list):
        sample_imgs_pred_t = np.copy(xt_prediction_list[selected_time - obs_dim].cpu().detach().numpy())[sample_id]
        
        ax1 = fig.add_subplot(gs[2, i + 2])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        ax1.set_title('Prediction', fontsize=30)
        plt.axis('off')
        plt.imshow(sample_imgs_pred_t.transpose((1, 2, 0)))
    
    plt.savefig('./jm/exp_oracle_mem_eval.png')
    plt.close()