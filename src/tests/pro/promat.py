import numpy as np

def pro_control_model(X_jk, elig_trace_jk, V_i, S_i, C_n, response_flag, resp_sig, threshold, dt, outDelay_spec, gamble_results, unobtained_outcomes, activ, div_r_i, gammaTD, alphaTD, w_baseline, rate_param, rate, noise, noise_factor, W_F_ni, W_C_ij, W_I_ij, respN, D_j, U_ijk):
    # Update the time steps
    elig_trace_jk = X_jk + 0.95 * elig_trace_jk
    V_i = np.dot(X_jk, U_ijk.T)  # Get the vector temporal prediction for this time step
    S_i = np.dot(D_j, W_S_ij)  # Get prediction of RO conjunctions

    # 10/10/19 - for visualization over t
    elig_trace_jk_t = X_jk + 0.95 * elig_trace_jk
    V_i_t = np.dot(X_jk, U_ijk.T)
    S_i_t = np.dot(D_j, W_S_ij)

    # Check for response occurrence
    if (np.max(C_n) > threshold and response_flag == 0) or (resp_sig and response_flag == 0):
        # then interpret the response
        response_flag = 1  # We have a response!
        learn_flag = 1  # and we should start learning something
        output = np.where(C_n == np.max(C_n))[0]  # What response did we make?

        if len(output) == 2:
            output = np.where(EV == np.max(EV))[0]  # correction: if same C, assume choice is for higher EV option
            if len(output) == 2:
                output = np.random.randint(2)  # correction: if still same, randomly choose option

        act_out[output] = 1  # register which output actually occurred.
        reaction_time = t  # keeps track of reaction time

        # 10/10/19 - for visualization
        response_flag_t[t] = 1

    # outcome is revealed after 750ms after decision (anticipatory period)
    if response_flag and t == (reaction_time + outDelay_spec / dt):
        outcome_flag = 1
        outcome_time = t

    if response_flag and outcome_flag:
        obtained_outcome = gamble_results[output]
        unobtained_outcome = unobtained_outcomes[output]
        tok_idx = obtained_outcome + 3
        act_idx_start = outN * (output - 1) + (tok_idx - 1) * (numbin - overlapbin) + 1
        O_i[act_idx_start:act_idx_start + numbin - 1] = activ  # 12 possible outcomes
        idx_count = 0  # set count to 0 to indicate beginning of outcome presentation to model.

        O_i_t[t] = O_i
        count_t[t] = 0

    # outcomes are interpreted as lasting 20 model iterations (200ms)
    if idx_count < 20:
        r_i = O_i / div_r_i
        on = 1
        r_i_t[t] = O_i / div_r_i
    else:
        r_i = O_i * 0
        on = 0
        r_i_t[t] = O_i * 0  # 10/10/19

    delta_i = r_i + gammaTD * V_i - prev_V_i  # Get TD prediction error, delta (Eq.7)
    U_ijk = U_ijk + alphaTD * np.dot(delta_i.T, prev_elig_trace)  # Update temporal prediction weights, U (Eq. 9)
    U_ijk[U_ijk < 0] = 0  # Rectify: no prediction weights under 0

    # Divide signal into positive and negative components
    omegaP = r_i * on - V_i  # Eq.15
    omegaP_nonzero = omegaP + w_baseline  # 7/7/15 JWB -- add a constant
    omegaN = V_i - O_i * on  # Eq.16

    omegaP[omegaP < 0] = 0  # Rectify negative components
    omegaN[omegaN < 0] = 0
    omegaP_nonzero[omegaP_nonzero < 0] = 0

    # for visualization
    delta_i_t[t] = r_i + gammaTD * V_i - prev_V_i
    omegaP_time[t] = r_i * on - V_i
    omegaP_nonzero_time[t] = w_baseline + (r_i * on - V_i)
    omegaN_time[t] = w_baseline + V_i - r_i * on
    # Rectify negative components
    omegaP_time[omegaP_time < 0] = 0
    omegaP_nonzero_time[omegaP_nonzero_time < 0] = 0
    omegaN_time[omegaN_time < 0] = 0

    # Prepare for the next trial iteration
    # calculate excitatory and inhibitory input to control units
    P_i = sigact(S_i)
    P_i_t[t] = P_i
    cont_sig_neg = -np.min([np.sum(P_i[:outN]), np.sum(P_i[outN:])] * W_F_ni + gamm, 0)
    cont_sig_pos = np.max([np.sum(P_i[:outN]), np.sum(P_i[outN:])] * W_F_ni - gamm, 0)

    E_i = rho * (np.dot(D_j, W_C_ij) + cont_sig_neg)  # net excitation to response units (Eq. 12)
    I_i = psi * (np.dot(C_n, W_I_ij)) + phi * (cont_sig_pos)  # net inhibition to response units (Eq. 13)

    # Update response unit activities
    C_n = C_n + (rate_param) * rate * dt * ((1 - C_n) * E_i - (C_n + 0.05) * (I_i + 1)) + np.random.randn(1, respN) * noise * np.sqrt(rate_param) * (noise_factor)  # (Eq. 11)
    C_n[C_n < 0] = 0  # Rectify negative elements
    omegaN_cat[:, t + 1] = omegaN
    omegaP_cat[:, t + 1] = omegaP
    omegaP_nonzero_cat[:, t + 1] = omegaP_nonzero  # revised model
    C_n_t[t] = C_n

    # temporary debugging:
    if False:
        tmp1 = omegaP_nonzero_cat.shape - omegaP_cat.shape
        if np.sum(tmp1 ** 2) > 0:
            raise ValueError('size mismatch in variables!')
        tmp2 = omegaN_nonzero_cat.shape - omegaN_cat.shape
        if np.sum(tmp2 ** 2) > 0:
            raise ValueError('size mismatch in variables!')

    # bookkeeping for TD model (t)
    prev_V_i = V_i
    prev_X_jk = X_jk
    prev_elig_trace = elig_trace_jk
    idx_count = idx_count + 1
    count_t[t + 1] = count_t[t] + 1

    return (elig_trace_jk, V_i, S_i, elig_trace_jk_t, V_i_t, S_i_t, response_flag, learn_flag, output, act_out, reaction_time, response_flag_t, outcome_flag, outcome_time, O_i, O_i_t, count_t, r_i, r_i_t, delta_i, delta_i_t, omegaP, omegaP_nonzero, omegaN, omegaP_time, omegaP_nonzero_time, omegaN_time, P_i, P_i_t, E_i, I_i, C_n, C_n_t, omegaN_cat, omegaP_cat, omegaP_nonzero_cat, prev_V_i, prev_X_jk, prev_elig_trace, idx_count)