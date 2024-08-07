import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
torch.set_default_dtype(torch.float64)
device = 'cuda'


def f_function(x, policy):
    '''
    upper-level function f = 0.5*sum(x*x) + 0.5*sum(policy*policy)
    '''
    return 0.5*torch.sum(x*x) + 0.5*torch.sum(policy*policy)

def f_grad_x(x, policy):
    return x

def f_grad_pi(x, policy):
    return policy

class RandMDP():
    '''
    lower-level MDP in the bilevel RL problem
    '''
    def __init__(self, state_space,action_space, args,tau=1, epsilon=1e-6, x_dim=100):
        self.args = args
        self.epsilon = epsilon
        self.gamma = args.gamma
        self.tau = tau
        self.state_space = state_space
        self.action_space = action_space
        self.x_dim = x_dim
        self.transition_probs = self.generate_transition_probs()

        self.x = torch.randn((x_dim,1), requires_grad=True, device=device)
        self.mat = torch.randn((x_dim,x_dim), device=device)
        self.rewards_coef = torch.zeros((self.state_space, self.action_space), device=device)
        for s in range(self.state_space):
            for a in range(self.action_space):
                Usa = np.random.uniform(0, 1)
                Us = np.random.uniform(0, 1)
                self.rewards_coef[s,a]= Usa * Us

        # self.rewards = self.generate_rewards()
        self.U = self.generate_U()

        

    
    def generate_U(self):
        U = torch.zeros((self.state_space, self.action_space, self.state_space),device=device)
        for s in range(self.state_space):
            for a in range(self.action_space):
                for sp in range(self.state_space):
                    if s==sp:
                        U[s,a,sp] = 1 - self.gamma*self.transition_probs[s,a,sp]
                    else:
                        U[s,a,sp] = -self.gamma*self.transition_probs[s,a,sp]
        return U

    def generate_transition_probs(self):
        transition_probs = torch.zeros((self.state_space, self.action_space, self.state_space))
        for s in range(self.state_space):
            for a in range(self.action_space):
                num_transitions = np.random.randint(1, self.state_space)
                Ssa = np.random.choice(range(self.state_space), size=num_transitions, replace=False)
                random_probs = np.random.random(num_transitions)
                random_probs /= random_probs.sum()  
                transition_probs[s, a, Ssa] = torch.tensor(random_probs)
        return transition_probs.to(device)

    def generate_rewards(self):
        rewards = torch.zeros((self.state_space, self.action_space),device=device)
        # self.rewards_grad = torch.zeros((self.state_space, self.action_space, self.n), device=device)
        for s in range(self.state_space):
            for a in range(self.action_space):
                rewards[s, a] = self.rewards_coef[s,a] * self.x.T@self.mat@self.x
                # debug
                # grad_zx = torch.autograd.grad(outputs=rewards[s, a], inputs=self.x)[0]
                # self.rewards_grad[s,a,:] = (torch.autograd.grad(outputs=rewards[s, a], inputs=self.x, retain_graph=True)[0]).squeeze()
        return rewards
    
    
    def soft_value_iteration(self, V, N=20):
        epsilon = self.epsilon
        max_iterations = N
        values = V.reshape(-1)

        with torch.no_grad():
            for i in range(max_iterations):
                prev_values = values.clone()
                # shape: S*A 
                temp = torch.exp( (1/self.tau) * (self.rewards + self.gamma*torch.einsum('saj,j->sa',self.transition_probs, prev_values)) )
                # shape: S*1       
                values = self.tau * torch.log(temp@torch.ones((self.action_space,1),device=device).squeeze())
                diff = torch.max(torch.abs(values - prev_values))
                # print(diff)
                if diff< epsilon:
                    break

            # shape: S
            soft_v = values
            # shape: S*A
            soft_q = self.rewards + self.gamma * torch.einsum('saj,j->sa',self.transition_probs, soft_v)
            soft_policy = torch.exp((1/self.tau) * (soft_q - soft_v.reshape((-1,1))))   

        return soft_v.reshape(-1,1), soft_q, soft_policy
    
    def hyper_grad_esti(self,x,policy,V,w):
        first_term = f_grad_x(x,policy)


        temp = torch.sum(self.rewards * (policy * f_grad_pi(x, policy)).detach())
        temp.backward(retain_graph=True)
        x_grad = x.grad.clone()
        x.grad.zero_()
        second_term = 1/self.tau * x_grad

        phi_x = self.compute_phi_x(V)
        third_term =  -1/self.tau * phi_x.T@w
        
        return first_term + second_term + third_term

    def compute_P_pi(self,policy):
        return torch.einsum('saj,sa->sj',self.transition_probs, policy)

    def compute_phi_x(self, V):
        phi_x = torch.zeros((self.state_space,self.x_dim),device=device)
        with torch.no_grad():
            # shape: S*A 
            temp = torch.exp( (1/self.tau) * (self.rewards + self.gamma*torch.einsum('saj,j->sa',self.transition_probs,V.reshape(-1))) )
            # shape: S*1       
            fenmu = temp@torch.ones((self.action_space,1),device=device)
        # shape S*1
        fenzi = torch.einsum('sa,sa->s',self.rewards,temp).reshape(-1,1)
        temp_phi = fenzi/fenmu
        
        for s in range(self.state_space):
            temp_grad = torch.autograd.grad(outputs=temp_phi[s,0], inputs=self.x, retain_graph=True)[0]
            phi_x[s,:] = temp_grad.squeeze()
        
        return phi_x




    def M_SoBiRL(self):
        args = self.args
        run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

        # TRY NOT TO MODIFY: seeding
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.tensorboard.patch(save=False)
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        global_step = 0       
        K = 3000
        beta = 8e-3
        zeta = 0.5

        w = torch.randn((self.state_space,1), device=device)   
        V = torch.zeros((self.state_space, 1)).to(device)         
        Q = self.gamma * torch.einsum('saj,j->sa',self.transition_probs, V.reshape(-1))
        policy = torch.exp((1/self.tau) * (Q - V.reshape((-1,1))))   
        policy = torch.nn.functional.softmax(policy, dim=1)
        self.rewards = self.generate_rewards()

        for k in range(K):
            global_step += 1

            A = torch.eye(self.state_space, device=device) - self.gamma*self.compute_P_pi(policy)
            A = A.T
            
            pi_nabla = policy * f_grad_pi(self.x,policy)         # S*A
            b = torch.einsum('saj,sa->j',self.U,pi_nabla)   # shape?
            b = b.reshape(-1,1)
            
            w = (w-zeta*(A.T@(A@w)-A.T@b)).detach()
            
            hyper_grad = self.hyper_grad_esti(self.x,policy,V,w)
            
            with torch.no_grad():
                self.x -= beta * hyper_grad

            self.rewards = self.generate_rewards()
            with torch.no_grad():
                V, Q, policy = self.soft_value_iteration(V, 10)
            V = V.detach()
            policy = policy.detach()

            print('global_step:',global_step,'---------------------------------')
            print('hyper_grad:',hyper_grad.norm())


            writer.add_scalar("hyper_grad", hyper_grad.norm(), global_step)
            writer.add_scalar("errors/value_errors", f_function(self.x,policy), global_step)

            

        writer.close()

