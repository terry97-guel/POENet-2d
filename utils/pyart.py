import torch
# def expm(vector):
#     if vector.shape[0] == 6:
#         return to_SE3(vector)
#     if vector.shape[0] == 4:
#         return to_SO3(vector)

def skew(p):
    device = p.device
    if len(p.size()) == 2:
        skew_p = torch.zeros(3,3,dtype=torch.float).to(device)
        p_temp = p

        skew_p[0,:] = torch.tensor([torch.tensor(0.0), -p_temp[2], p_temp[1]])
        skew_p[1,:] = torch.tensor([p_temp[2], torch.tensor(0.0), -p_temp[0]])
        skew_p[2,:] = torch.tensor([-p_temp[1], p_temp[0],torch.tensor(0.0)])

    elif len(p.size()) == 3:
        num_p = p.size()[0]
        skew_p = torch.zeros(num_p,3,3,dtype=torch.float).to(device)
    
        for i in range(num_p):
            p_temp = p[i]

            skew_p[i,0,:] = torch.tensor([torch.tensor(0.0), -p_temp[2], p_temp[1]])
            skew_p[i,1,:] = torch.tensor([p_temp[2], torch.tensor(0.0), -p_temp[0]])
            skew_p[i,2,:] = torch.tensor([-p_temp[1], p_temp[0],torch.tensor(0.0)])
    else:
        raise(Exception("demension for p is invalid"))

    return skew_p

def rpy2r(rpy):
    device = rpy.deivce
    if len(rpy.size()) == 2:
        assert(rpy.size()[0] == 3)
        R = torch.zeros(3,3).to(device)
        num_rpy = 1

        rpy_temp = rpy
        r = rpy_temp[0]
        p = rpy_temp[1]
        y = rpy_temp[2]

        R[0,:] = torch.tensor([
            torch.cos(y)*torch.cos(p),
            -torch.sin(y)*torch.cos(r) + torch.cos(y)*torch.sin(p)*torch.sin(r),
            torch.sin(y)*torch.sin(r)+torch.cos(y)*torch.sin(p)*torch.cos(r)
            ])
        
        R[1,:] = torch.tensor([
            torch.sin(y)*torch.cos(p),
            torch.cos(y)*torch.cos(r) + torch.sin(y)*torch.sin(p)*torch.sin(r),
            -torch.cos(y)*torch.sin(r)+torch.sin(y)*torch.sin(p)*torch.cos(r)
            ])

        R[2,:] = torch.tensor([
            -torch.sin(p),
            torch.cos(p)*torch.sin(r),
            torch.cos(p)*torch.cos(r)
            ])
    elif len(rpy.size()) == 3:
        assert(rpy.size()[1] == 3)
        num_rpy = rpy.size()[0]
        R = torch.zeros(num_rpy,3,3)
        
        for i in range(num_rpy):
            rpy_temp = rpy[i]
            r = rpy_temp[0]
            p = rpy_temp[1]
            y = rpy_temp[2]

            R[i,0,:] = torch.tensor([
                torch.cos(y)*torch.cos(p),
                -torch.sin(y)*torch.cos(r) + torch.cos(y)*torch.sin(p)*torch.sin(r),
                torch.sin(y)*torch.sin(r)+torch.cos(y)*torch.sin(p)*torch.cos(r)
                ])
            
            R[i,1,:] = torch.tensor([
                torch.sin(y)*torch.cos(p),
                torch.cos(y)*torch.cos(r) + torch.sin(y)*torch.sin(p)*torch.sin(r),
                -torch.cos(y)*torch.sin(r)+torch.sin(y)*torch.sin(p)*torch.cos(r)
                ])

            R[i,2,:] = torch.tensor([
                -torch.sin(p),
                torch.cos(p)*torch.sin(r),
                torch.cos(p)*torch.cos(r)
                ])
    else:
        raise(Exception("demension for rpy is invalid"))
    
    return R

def pr2t(p,r):
    device = p.device
    assert(len(p.size()) == len(r.size()))
    if len(p.size()) == 2:
        assert(p.size()[0] == 3 & r.size()[0] == 3 & r.size()[0] == 3)
        T = torch.zeros(4,4,dtype=torch.float).to(device)

        T[0:3,0:3] = r
        T[0:3,3] =  p.squeeze()
        T[3,3] = 1
    elif len(p.size()) == 3:
        assert(p.size()[1] == 3 & r.size()[1] == 3 & r.size()[1] == 3)
        num_p = p.size()[0]
        T = torch.zeros(num_p,4,4,dtype=torch.float).to(device)

        for i in range(num_p):
            T[i,0:3,0:3] = r[i]
            T[i,0:3,3] =  p[i].squeeze()
            T[i,3,3] = 1
    else:
        raise(Exception("demension for position or orientation is invalid"))
    return T

def POE(twist, q_value, verbose =False): #number of set of twist is one & number of q_value is n_joint
    eps = 1e-10
    device = twist.device

    assert(twist.size()[1] == 6)

    if len(q_value.size()) == 1:
        #PLZ Fill me
        pass

    elif len(q_value.size()) == 2:                      #q_value.size() = [batch_size, n_joint] #twist.size() = [n_joint,6,1]
        assert(twist.size()[0] == q_value.size()[1])    #number of n_joint should be same
        batch_size = q_value.size()[0]
        n_joint = twist.size()[0]
        T = torch.zeros(batch_size,4,4).to(device)
        
        for batch in range(batch_size):
            poe = torch.eye(4).to(device)
            for joint in range(n_joint):
                T_temp = torch.zeros(4,4,dtype = torch.float).to(device)
                w = twist[joint,0:3,0].view(3,1)
                v = twist[joint,3:6,0].view(3,1)
                if torch.norm(w) < eps:
                    if torch.norm(v) <  eps:
                        theta = 1
                    else:
                        theta = torch.norm(w)
                elif abs(torch.norm(w) - 1) > eps:
                    if verbose:
                        print("Warning: [POE] >> joint twist not normalized")
                    theta = torch.norm(w)
                else:
                    theta = 1
                w = w/theta
                v = v/theta
                q = q_value[batch,joint] * theta
                
                A = rodrigues(w,q)
                B = (q*torch.eye(3).to(device)+(1-torch.cos(q))*skew(w) + (q-torch.sin(q))*skew(w)@skew(w)) @ v
                T_temp[0:3,0:3] = A
                T_temp[0:3,3] = B.squeeze()
                T_temp[3,3] = 1
                poe = poe @ T_temp
            T[batch] = poe
    

    else:
        raise(Exception("demension for twist & q_value doesn't match"))

    return T


def rodrigues(w,q,verbose = False):
    eps = 1e-10
    device = w.device

    if torch.norm(w) < eps:
        R = torch.eye(3).to(device)
        return R
    if abs(torch.norm(w)-1) > eps:
        if verbose:
            print("Warning: [rodirgues] >> joint twist not normalized")
    
    theta = torch.norm(w)
    w = w/theta
    q = q*theta

    w_skew = skew(w)
    R = torch.eye(3).to(device) + w_skew * torch.sin(q) + w_skew @ w_skew * (1-torch.cos(q))

    return R