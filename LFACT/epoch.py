import time
import numpy as np
from reader4 import ptb_iterator

def run_epoch(config,session, m, data, index, eval_op, fmea_cal=False, max_steps=None, verbose=False):
    """Runs the model on the given data."""
    #epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    num_batch_steps_completed = 0
    bpc = []
    Nt_all=[]

    for step, (x) in enumerate(ptb_iterator(data, index, m.batch_size, m.num_steps, config)):

        cost, Nt, state,_ = session.run([m.cost, m.Nt, m.final_state, eval_op],
                                     {m.input_data: x})
        costs += cost
        Nt_all.append(Nt)

        iters +=1# m.num_steps
        num_batch_steps_completed += 1
        
        if fmea_cal:
            state = np.reshape(state,[config.batch_size*config.num_steps,config.classes])
            result = np.zeros([config.batch_size*config.num_steps])
            for i in range(config.batch_size*config.num_steps):
                result[i] = np.argmax(state[i]) #np.where(state[i,j] == np.amax(state[i,j]))[0][0]
                #result[i,np.argmax(state[i])]=1
                    
            yy=x[:,1:,:]
            yy=np.reshape(yy,[config.batch_size*config.num_steps,config.classes])
            
            lab = np.zeros([config.batch_size*config.num_steps])
            for i in range(config.batch_size*config.num_steps):
                lab[i] = np.argmax(yy[i])
            
            res2 = lab - result
            res2[res2!=0]=1
            
            bpc.append(1-res2)
            
    '''
    Nt_all = np.array(Nt_all)
    Nt_all = Nt_all.T
    Ntmax = np.max(Nt_all, axis=1)
    Nt95 = np.percentile(Nt_all,95,axis=1)
    F=open('seqACT_Nt.txt','w') 
    F.writelines('\t'.join(str(j) for j in i) + '\n' for i in Ntmax)
    F.writelines('\t'.join(str(j) for j in i) + '\n' for i in Nt95)
    F.close()      
    '''
    Nt_all=np.reshape(Nt_all,[-1,config.num_steps])
    
    bpc=np.squeeze(bpc)
    return (costs / iters), bpc,Nt_all
