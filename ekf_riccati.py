import numpy as np
from observation_operators import alt_obs
from observation_operators import obs_seq
from l96 import l96_step_TLM
from l96 import l96_rk4_step
import copy

########################################################################################################################
# forecast ricatti equation


def ekf_fore(M, P, Q, R, H, I, inflation=1.0):
    """"This function returns a sequence of analysis error covariance matrices for the ricatti equation"""

    # the forecast riccati is given in the square root form
    [U, s, V] = np.linalg.svd(P)
    X = U @ np.diag(np.sqrt(s))

    Omega = H.T @ np.linalg.inv(R) @ H
    XOX = X.T @ Omega @ X

    P = M @ X @ np.linalg.inv(I + XOX) @ X.T @ M.T 
    P = P * inflation + Q

    return P


########################################################################################################################
# simulate


def simulate_ekf_jump(x, truth, obs_dim, h, f, tanl, tl_fore_steps, Q, obs_un, seed, burn, infl=1.0):
    """This function simulates the extended kalman filter with alternating obs operator"""

    # define initial parameters
    [sys_dim, nanl] = np.shape(truth)

    # define the observations
    H = alt_obs(sys_dim, obs_dim)
    obs_seed = seed + 10000
    obs = obs_seq(truth, obs_dim, obs_un, H, obs_seed)
    P = np.eye(sys_dim)
    Q = Q * tanl**2
    R = np.eye(obs_dim) * obs_un
    I = np.eye(sys_dim)

    x_fore = np.zeros([sys_dim, nanl])
    x_anal = np.zeros([sys_dim, nanl])

    for i in range(nanl):
        # re-initialize the tangent linear model
        Y = np.eye(sys_dim)

        # make a forecast along the non-linear trajectory and generate the tangent model, looping in the integration
        # steps of the tangent linear model
        for j in range(tl_fore_steps):
            [x, Y] = l96_step_TLM(x, Y, h, f, l96_rk4_step)

        # define the forecast state and uncertainty
        x_fore[:, i] = x
        P = ekf_fore(Y, P, Q, R, H, I, inflation=infl)

        # define the kalman gain and perform analysis
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        x = x + K @ (obs[:, i] - H @ x)
        x_anal[:, i] = x

    # compute the rmse for the forecast and analysis
    fore = x_fore - truth
    fore = np.sqrt(np.mean(fore * fore, axis=0))

    anal = x_anal - truth
    anal = np.sqrt(np.mean(anal * anal, axis=0))

    f_rmse = []
    a_rmse = []

    for i in range(burn + 1, nanl):
        f_rmse.append(np.mean(fore[:i]))
        a_rmse.append(np.mean(anal[:i]))

    return {'fore': f_rmse, 'anal': a_rmse}

########################################################################################################################
