import numpy as np
from observation_operators import alt_obs
from observation_operators import obs_seq
from l96 import l96_rk4_step
from l96 import l96_step_TLM
from l96 import l96_rk4_stepV
import copy


########################################################################################################################
# EKF-AUS full forecast error Riccati equation


def EKFAUS_fore(P_0, B_0, T_1, H_0, Q_ff, R, inflation=1.0):
    """"This function returns the estimated forecast error covariance matrix of EKF-AUS

    P_0 is current prior error covariance, B_0 is a matrix of the BLVs corresponding to the filtered subspace,
    T_1 is the upper triangular matrix for the dynamics in the leading BLVs, H_0 is the observation operator at current
    time, Q_ff is the model error covariance projected into the filtered space, R is the observational error
    covariance, inflation is the multiplicative inflation factor.
    """

    # infer parameters
    [sys_dim, ens_dim] = np.shape(B_0)
    H_b = H_0 @ B_0

    # define the identity matrix of dimension of the filtered space
    I_f = np.eye(ens_dim)

    # define the square root of the filtered variables for the update
    [U, S_0, V] = np.linalg.svd(P_0)
    S_0 = U @ np.diag(np.sqrt(S_0)) @ U.T

    # precision matrix
    Omega = S_0 @ H_b.T @ np.linalg.inv(R) @ H_b @ S_0

    # square root forecast error covariance
    P_1 = T_1 @ S_0 @ np.linalg.inv(I_f + Omega) @ S_0 @ T_1.T 
    P_1 = P_1 * inflation + Q_ff

    return P_1

########################################################################################################################
# KFAUSE full forecast error Riccati equation


def KFAUSE_fore(P_0, B_0, T_1, H_0, ens_dim, Q_1, R):
    """"This function returns the analytical forecast error covariance of KF-AUSE
    
    P_0 is current prior error covariance written in a basis of BLVs, B_0 is a matrix of the BLVs, T_1 is the upper 
    triangular matrix for the dynamics in the BLVs, H_0 is the observation operator at current time, ensdim is the 
    dimension of the filtered subspace, Q_1 is the model error covariance, R is the observational error covariance.
    """

    # infer parameters define storage
    [obs_dim, sys_dim] = np.shape(H_0)
    unfil_d = sys_dim - ens_dim
    I_f = np.eye(ens_dim)
    I_u = np.eye(unfil_d)

    # break P into its sub components
    P_0uu = np.squeeze(P_0[ens_dim:, ens_dim:])
    P_0fu = P_0[:ens_dim, ens_dim:]
    S_0sqr = P_0[:ens_dim, :ens_dim]

    # define the span of the filtered and unfiltered variables
    B_0f = B_0[:, :ens_dim]
    B_0u = B_0[:, ens_dim:]

    # and the observation operator on their spans
    H_f = H_0 @ B_0f
    H_u = H_0 @ B_0u

    # the upper triangular matrix is restricted to sub-blocks
    T_1pp = T_1[:ens_dim, :ens_dim]
    T_1pm = T_1[:ens_dim, ens_dim:]
    T_1mm = T_1[ens_dim:, ens_dim:]

    # as is the model error matrix, assumed to be written in a basis of BLVs
    Q_ff = Q_1[:ens_dim, :ens_dim]
    Q_fu = Q_1[:ens_dim, ens_dim:]
    Q_uu = Q_1[ens_dim:, ens_dim:]

    # define the square root of the filtered variables for the update
    [U, S_0, V] = np.linalg.svd(S_0sqr)
    S_0 = U @ np.diag(np.sqrt(S_0)) @ U.T

    # precision matrix
    Omega0 = S_0 @ H_f.T @ np.linalg.inv(R) @ H_f @ S_0

    # reduced gain equation
    J_0 = S_0sqr @ H_f.T @ np.linalg.inv(H_f @ S_0sqr @ H_f.T + R)

    # UPDATE STEPS

    # filtered block update
    S_1sqr = T_1pp @ S_0 @ np.linalg.inv(I_f + Omega0) @ S_0 @ T_1pp.T + Q_ff

    if unfil_d == 1:
        tmp = (T_1pm - np.reshape(T_1pp @ J_0 @ H_u, [ens_dim, 1]))
        sigma = P_0uu * tmp @ tmp.T
        Phi = T_1pp @ (I_f - J_0 @ H_f) @ P_0fu
        # reshape to get the exterior product of the two vectors and prevent unintended broadcasting
        Phi = np.reshape(Phi, [ens_dim, 1]) @ (T_1pm - np.reshape(T_1pp @ J_0 @ H_u, [ens_dim, 1])).T

        sigma += Phi + Phi.T
        S_1sqr += sigma
    else:
        sigma = (T_1pm - T_1pp @ J_0 @ H_u) @ P_0uu @ (T_1pm - T_1pp @ J_0 @ H_u).T
        Phi = T_1pp @ (I_f - J_0 @ H_f) @ P_0fu @ (T_1pm - T_1pp @ J_0 @ H_u).T
        sigma += Phi + Phi.T
        S_1sqr += sigma

    # unfiltered update step
    if unfil_d == 1:
        P_1uu = Q_uu + P_0uu * T_1mm**2
    else:
        P_1uu = Q_uu + T_1mm @ P_0uu @ T_1mm.T

    # cross covariance steps
    if unfil_d == 1:
        P_1fu = (T_1pm - np.reshape(T_1pp @ J_0 @ H_u, [ens_dim, 1])) * P_0uu * T_1mm + Q_fu
        P_1fu += np.reshape(T_1pp @ (I_f - J_0 @ H_f) @ P_0fu * T_1mm, [ens_dim, 1])
    else:
        P_1fu = (T_1pm - T_1pp @ J_0 @ H_u) @ P_0uu @ T_1mm.T + Q_fu
        P_1fu += T_1pp @ (I_f - J_0 @ H_f) @ P_0fu @ T_1mm.T

    # broadcast the updates into the matrix P_1
    P_1 = np.zeros([sys_dim, sys_dim])
    P_1[:ens_dim, :ens_dim] = S_1sqr 
    P_1[:ens_dim, ens_dim:] = P_1fu
    P_1[ens_dim:, :ens_dim] = P_1fu.T
    P_1[ens_dim:, ens_dim:] = P_1uu

    return P_1


########################################################################################################################

def simulate_ekf_aus_jump(x_0, truth, ens_dim, obs_dim, h, f, tanl, tl_fore_steps, Q, obs_un, seed, burn, m_inf=1.0):
    """This function simulates the extended kalman AUS filter with alternating obs operator
    
    x_0 is the initial state for the filter, truth is the full sequence of truth states at observation times,
    ens_dim is the dimension of the filtered subspace, obs_dim is the dimension of the observations,
    h is the tangent-linear model integration step size (nonlinear trajectory is half), f is the L96 forcing parameter,
    tanl is the continuous time between observations, tl_fore_steps is the number of tangent linear model integration
    steps between observations, Q is the full model error covariance matrix, obs_un is the observational error
    variance, seed is the random seed used for the experimental initialization, burn is the number of analyses to
    discard before computing the RMSE, m_infl is the scale of the multiplicative inflation factor set to 1.0 as
    default.
    """

    # infer initial parameters and define storage
    [sys_dim, nanl] = np.shape(truth)
    lle = np.zeros(ens_dim)

    # define the observation operator
    H = alt_obs(sys_dim, obs_dim)
    
    # re-seed the observation operator as a function of, but different than, the initial seed
    obs_seed = seed + 10000
    obs = obs_seq(truth, obs_dim, obs_un, H, obs_seed)

    # define the initial error covariances with model error scaled by the length between observations squared
    R = np.eye(obs_dim) * obs_un
    P_0 = np.eye(ens_dim)
    Q = Q * tanl**2
    
    # define the initialization for the leading BLVs, i.e., the span of the filtered space
    B_0 = np.eye(sys_dim, M=ens_dim)

    x_fore = np.zeros([sys_dim, nanl])
    x_anal = np.zeros([sys_dim, nanl])

    for i in range(nanl):
        # we re-initialize the tangent linear model with the last computed BLVs and make a forecast along the non-linear
        # trajectory and generate the tangent model, looping in the integration steps of the tangent linear model

        # initialize the perturbations in the tl model
        B = B_0
        x = x_0
        for j in range(tl_fore_steps):
            [x, B] = l96_step_TLM(x, B, h, f, l96_rk4_step)

        # define the forecast state
        x_1 = x
        x_fore[:, i] = x_1

        # compute the backward Lyapunov vectors at the next step recursively
        B_1, T_1 = np.linalg.qr(B)
        lle += np.log(np.abs(np.diagonal(T_1)))

        # we write Q in the basis of BLV
        Q_tmp = B_1.T @ Q @ B_1

        # the AUS Riccati equation is defined in terms of the above values
        P_1 = EKFAUS_fore(P_0, B_0, T_1, H, Q_tmp, R, inflation=m_inf)

        # update the ensemble span of the leading BLVs and the associated observation operator for the analysis time
        # in order to compute the analysis state from the forecast
        H_b = H.dot(B_1)

        # define the kalman gain and find the analysis mean with the new forecast uncertainty
        K = np.linalg.inv(H_b @ P_1 @ H_b.T + R)
        K = B_1 @ P_1 @ H_b.T @ K
        x_1 = x_1 + K @ (obs[:, i] - H @ x_1)
        x_anal[:, i] = x_1

        # re-initialize quantities for recursive computation
        x_0 = x_1
        P_0 = P_1
        B_0 = B_1

    # compute the rmse for the forecast and analysis
    fore = x_fore - truth
    fore = np.sqrt(np.mean(fore * fore, axis=0))

    anal = x_anal - truth
    anal = np.sqrt(np.mean(anal * anal, axis=0))

    f_rmse = []
    a_rmse = []

    for i in range(burn + 1, nanl):
        f_rmse.append(np.mean(fore[burn: i]))
        a_rmse.append(np.mean(anal[burn: i]))

    le = lle/(nanl * tl_fore_steps * h)

    return {'fore': f_rmse, 'anal': a_rmse, 'le': le}


########################################################################################################################

def simulate_kf_ause_jump(x_0, truth, ens_dim, obs_dim, h, f, tanl, tl_fore_steps, Q, obs_un, seed, burn):
    """This function simulates the extended kalman filter aus exact (EKF-AUSE) with alternating obs operator
    
    x_0 is the initial state for the filter, truth is the full sequence of truth states at observation times,
    ens_dim is the dimension of the filtered subspace, obs_dim is the dimension of the observations,
    h is the tangent-linear model integration step size (nonlinear trajectory is half), f is the L96 forcing parameter,
    tanl is the continuous time between observations, tl_fore_steps is the number of tangent linear model integration
    steps between observations, Q is the full model error covariance matrix, obs_un is the observational error
    variance, seed is the random seed used for the experimental initialization, burn is the number of analyses to
    discard before computing the RMSE.
    """

    # infer initial parameters and define storage
    [sys_dim, nanl] = np.shape(truth)
    lle = np.zeros(sys_dim)

    # define the observations
    H = alt_obs(sys_dim, obs_dim)
    obs_seed = seed + 10000
    obs = obs_seq(truth, obs_dim, obs_un, H, obs_seed)
    P_0 = np.eye(sys_dim)
    R = np.eye(obs_dim) * obs_un
    Q = Q * tanl**2
    B_0 = np.eye(sys_dim)

    x_fore = np.zeros([sys_dim, nanl])
    x_anal = np.zeros([sys_dim, nanl])

    for i in range(nanl):
        # we re-initialize the tangent linear model with the last computed BLVs and make a forecast along the non-linear
        # trajectory and generate the tangent model, looping in the integration steps of the tangent linear model

        #initialize the tangent linear model
        x = x_0
        B = B_0

        for j in range(tl_fore_steps):
            [x, B] = l96_step_TLM(x, B, h, f, l96_rk4_step)

        # define the forecast state
        x_1 = x
        x_fore[:, i] = x_1

        # compute the backward Lyapunov vectors at the next step recursively
        B_1, T_1 = np.linalg.qr(B)
        lle += np.log(np.abs(np.diagonal(T_1)))

        # we write Q in the basis of BLV
        Q_tmp = B_1.T @ Q @ B_1

        # the AUSE Riccati equation is defined in terms of the above values
        P_1 = KFAUSE_fore(P_0, B_0, T_1, H, ens_dim, Q_tmp, R)

        # update the ensemble span of the leading BLVs and the associated observation operator for the analysis time
        B_f = B_1[:, :ens_dim]
        H_f = H.dot(B_f)

        # define the kalman gain and find the analysis mean with the new forecast uncertainty
        S_sqr = P_1[:ens_dim, :ens_dim]
        K = B_f @ S_sqr @ H_f.T @ np.linalg.inv(H_f @ S_sqr @ H_f.T + R)

        x_1 = x_1 + K @ (obs[:, i] - H @ x_1)
        x_anal[:, i] = x_1

        # re-initialize
        x_0 = x_1
        P_0 = P_1
        B_0 = B_1

    # compute the rmse for the forecast and analysis
    fore = x_fore - truth
    fore = np.sqrt(np.mean(fore * fore, axis=0))

    anal = x_anal - truth
    anal = np.sqrt(np.mean(anal * anal, axis=0))

    f_rmse = []
    a_rmse = []

    for i in range(burn + 1, nanl):
        f_rmse.append(np.mean(fore[burn: i]))
        a_rmse.append(np.mean(anal[burn: i]))

    le = lle/(nanl * tl_fore_steps * h)

    return {'fore': f_rmse, 'anal': a_rmse, 'le': le}

########################################################################################################################
