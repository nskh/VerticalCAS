export relhs,dh0s,dh1s,taus,pas,nstates,actions, action_names

# State dimension cutpoints
const vels = vcat(range(-100,-60,length=5),range(-50,-35,length=4),range(-30,30,length=21),range(35,50,length=4),range(60,100,length=5))
const relhs   = vcat(range(-8000,-4000,length=5),range(-3000,-1250,length=8),range(-1000,-800,length=3),range(-700,-150,length=12),range(-100,100,length=9), range(150,700,length=12),range(800,1000,length=3),range(1250,3000,length=8),range(4000,8000,length=5))
const dh0s = vels
const dh1s = vels
const taus  = range(0,40,length=41)
const pas = [1,2,3,4,5,6,7,8,9]

# Number of states
const nstates = length(relhs)*length(dh0s)*length(taus)*length(dh1s)

# Actions
const actions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
const action_names = ["COC","DNC","DND","DES1500","CL1500","SDES1500","SCL1500","SDES2500","SCL2500"]