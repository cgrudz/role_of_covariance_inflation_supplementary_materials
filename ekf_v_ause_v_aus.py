import numpy as np
from l96 import l96_rk4_step
from ekf_riccati import simulate_ekf_jump
from ekf_ause_riccati import simulate_kf_ause_jump
from ekf_ause_riccati import simulate_ekf_aus_jump
from scipy.linalg import circulant
import pickle

########################################################################################################################


def experiment(args):
    """This experiment tests the perfomance of the EKF vs EKF-AUS w/o inflation vs EKF-AUSE 

    The experiment is functionalized to be run over different parameter configurations.  It takes arguments
    seed     --- initial random seed for generator
    tanl     --- continuous time between analyses
    model_er --- scaling of the model error covariance
    obs_un   --- scaling of the observational error covariance
    obs_dim  --- number of observations
    """


    ####################################################################################################################
    # Define static experiment parameters
    ####################################################################################################################

    [seed, tanl, model_er, obs_un, obs_dim] = args

    # state dimension
    sys_dim = 40

    # forcing for Lorenz 96
    f = 8

    # number of analyses
    nanl = 12

    # h = tangent linear model integration step size (nonlinear step size is half)
    h = .1
    nl_step = .5 * h

    # nonlinear integration steps between observations
    tl_fore_steps = int(tanl / h)
    nl_fore_steps = int(tanl / nl_step)

    # continous time of the spin period before experiment begins
    spin = 50

    # number of analyses to throw out before filter statistics are computed
    burn = 2

    # model error covariance
    Q = circulant([1, .5, .25] + [0]*(sys_dim - 5) + [.25, .5])
    Q = Q * model_er

    # initial value
    np.random.seed(seed)
    x = np.random.multivariate_normal(np.zeros(sys_dim), Q * sys_dim)

    ####################################################################################################################
    # generate the truth
    ####################################################################################################################

    # spin the model on to the attractor
    for i in range(int(spin / nl_step)):
        x = l96_rk4_step(x, nl_step, f)
        if (i % nl_fore_steps) == 0:
            # add jumps to the process
            x += np.random.multivariate_normal(np.zeros(sys_dim), Q) * tanl

    # the model and the truth will have the same initial condition
    x_init = x

    # generate the full length of the truth tajectory which we assimilate
    truth = np.zeros([sys_dim, nanl])

    for i in range(nanl):
        # integrate until the next observation time
        for j in range(nl_fore_steps):
            x = l96_rk4_step(x, nl_step, f)

        # jump at observation
        x += np.random.multivariate_normal(np.zeros(sys_dim), Q) * tanl
        truth[:, i] = x

    ####################################################################################################################
    # we simulate EKF for the set observations
    ####################################################################################################################

    ekf = simulate_ekf_jump(x_init, truth, obs_dim, h, f, tanl, tl_fore_steps, Q, obs_un, seed, burn)
    data = {'ekf': ekf}
    f_name = './data/EKF_rmse_seed_' + str(seed).zfill(3) + '_analint_' + str(tanl).zfill(3) + '_obs_un_' +\
             str(obs_un).zfill(3) + '_model_err_scale_' + str(model_er).zfill(3) + '_sys_dim_' + str(sys_dim).zfill(2)+\
             '_obs_dim_' + str(obs_dim) + '.txt'
    fid = open(f_name, 'wb')
    pickle.dump(data, fid)
    fid.close()

    ####################################################################################################################
    # simulate EKF-AUS/E
    ####################################################################################################################

    ens_range = np.arange(14, 40)
    ens_range = ens_range[::-1]

    for i in ens_range:
        # reset dictionary
        data = {}

        # aus
        key = 'aus'
        data[key] = simulate_ekf_aus_jump(x_init, truth, i, obs_dim, h, f, tanl, tl_fore_steps, Q, obs_un, seed, burn)

        # then runs of kf-ause
        key = 'ause'
        data[key] = simulate_kf_ause_jump(x_init, truth, i, obs_dim, h, f, tanl, tl_fore_steps, Q, obs_un, seed, burn)

        # open up a storage file
        f_name = './data/AUS_AUSE_rmse_seed_' + str(seed).zfill(3) + '_analint_' + str(tanl).zfill(3) + '_obs_un_' + \
                 str(obs_un).zfill(3) + '_model_err_scale_' + str(model_er).zfill(3) + '_sys_dim_' + \
                 str(sys_dim).zfill(2) + '_obs_dim_' + str(obs_dim) + '_ens_dim_' + str(i).zfill(2) + '.txt'

        fid = open(f_name, 'wb')
        pickle.dump(data, fid)
        fid.close()


########################################################################################################################

