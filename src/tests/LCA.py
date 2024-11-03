reward = 0.1

lr = 0.1 # Local Rate node
gr = 0.1 # Global Rate node

# value layer weights
w_rlr = 0.1  # reward -> local rate
w_lrlr = 0.1 # recurrent local rate
w_rgr = 0.1  # reward -> global rate
w_grgr = 0.1 # recurrent global rate

Er = 0.1

sd = 0.1 # Stay decision node
ld = 0.1 # Leave decision node

s = 0

# decision layer weights 
w_vs = 0.1 # value -> stay decision
w_vl = 0.1 # value -> leave decision
w_ss = 0.1 # stay decision recurrent
w_ll = 0.1 # leave decision recurrent
w_dc = -0.1 # weight for competing stay and leave decision

Ed = 0.1

decision = False

lr = -lr + w_rlr * reward + w_lrlr * lr + Er
gr = -gr + w_rgr * reward + w_grgr * gr + Er

if decision:
    s = 1
else:
    s = 0
sd = -sd + w_vs*s*lr + w_ss*sd + w_dc*ld + Ed
ld = -ld + w_vl*s*gr + w_ll*ld + w_dc*sd + Ed