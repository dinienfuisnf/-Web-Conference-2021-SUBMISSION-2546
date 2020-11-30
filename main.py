#!/usr/bin/python3
from layers import GNNLayer
import os, time
from tensorflow.keras.optimizers import Adam
from env_new import City_env
import tensorflow as tf
import argparse 
from copy import deepcopy
from get_args import *
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Layer, Dense, Activation, Flatten, Input, Concatenate, Reshape, Lambda, Multiply, \
    Permute, LSTM, RepeatVector
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
num_pop = args.num_pop
CYCLE_EPOCHS=2
num_his = 2
num_dim = 32
dim_state=9
N_CYCLES = 10000
LEARNING_RATE =1e-4
BATCH_SIZE = 32
MINIBATCH = 16
GAMMA = 0.99
EPSILON = 0.1
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
tf.config.experimental.set_memory_growth = True
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048*7)]
)
localtime = time.strftime("%m%d_%H:%M:%S", time.localtime())

class DiscretePPO:
    def __init__(self, V, pi):
        self.V = V()
        self.pi = pi()
        self.old_pi=pi()
        self.old_pi.set_weights(self.pi.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
        self.advantage = np.zeros((16, 1))
        self.vis=np.zeros((1,10000,2,12))
        self.vis_=np.zeros((16,10000,2,12))
        self.old_pre=np.zeros((16,10000,4))
        self.loss_clipping = 0.1
        self.entropy_loss = 0.1
    def pick_action(self, S,visit):
        S_ = deepcopy(S[ :, 3])
        S=S[np.newaxis, :]
        visit=visit[np.newaxis, :]
        action = self.pi.predict_on_batch([S, self.advantage,visit,self.old_pre])
        action=action.reshape(10000,4)
        action_re=deepcopy(action)
        action=np.cumsum(action,axis=-1)
        action_sampled = np.zeros((10000,4))
        S_=S_.reshape(10000,1)
        for i in range(10000):
            if S_[i]<action[i,0]:
                action_sampled[i,0]=1
            elif S_[i]<action[i,1]:
                action_sampled[i,1]=1
            elif S_[i]<action[i,2]:
                action_sampled[i,2]=1
            else:
                action_sampled[i,3]=1
        return action_sampled,action_re

    def train_minibatch(self, SARTS_minibatch):
        S,Vis,A, R, T, S2 = SARTS_minibatch
        A = A.reshape(16,10000,4)
        R = R.reshape(16, 1)
        T = T.reshape(16, 1)
        T=T+0
        next_V = self.V.predict_on_batch([S2,Vis])
        next_V=next_V*T
        target=R + GAMMA * next_V
        advantage = target - self.V.predict_on_batch([S,Vis])
        metrics_V=self.V.train_on_batch([S,Vis],target)
        print('metrics_V:',metrics_V)
        old_pi_pre=self.old_pi.predict_on_batch([S, self.advantage,Vis,self.old_pre])
        metrics_A=self.pi.train_on_batch([S, advantage,Vis,old_pi_pre],A)
        print('metrics_A:', metrics_A)

    def train(self, SARTS_batch):
        S,Vis,A, R, T, S2 = SARTS_batch

        for _ in range(CYCLE_EPOCHS):
            # shuffle and split into minibatches!
            shuffled_indices=np.arange(BATCH_SIZE)
            shuffled_indices_ = np.random.shuffle(shuffled_indices)
            num_mb = BATCH_SIZE // MINIBATCH
            for minibatch_indices in np.split(shuffled_indices, num_mb):
                mb_SARTS=[S[minibatch_indices],Vis[minibatch_indices],A[minibatch_indices],R[minibatch_indices],T[minibatch_indices],
                          S2[minibatch_indices]]
                self.train_minibatch(mb_SARTS)

        for old_pi_w, pi_w in zip(self.old_pi.weights, self.pi.weights):
            old_pi_w.assign(pi_w)
    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.pi.load_weights(actor_filepath)
        self.V.load_weights(critic_filepath)


    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.pi.save_weights(actor_filepath, overwrite=overwrite)
        self.V.save_weights(critic_filepath, overwrite=overwrite)
def train_PPO(agent, envs,rounds,path_):
    step=0
    step1=0
    s,visit = deepcopy(envs.reset())
    SARTS_samples = []
    done = False
    path = path_+ localtime
    os.makedirs(path)
    print('Save at', path)
    save_path = path + '/' + 'agent_weights.ckpt'
    for inter in range(rounds):

        a,a_ = agent.pick_action(s,visit)
        s2, r, done, _,aa,visit_ = envs.step(a)
        SARTS_samples.append((s,visit,a,r,done,s2))
        done=False
        aa=False
        if done or aa:
            s,visit = deepcopy(envs.reset())
        else:
            s = s2
            visit = visit_
        step+=1
        if (inter+1)%10==0:
            agent.save_weights(save_path, overwrite=True)
        if step==32:
            SARTS_batch = [np.stack(X, axis=0) for X in zip(*SARTS_samples)]
            agent.train(SARTS_batch)
            step=0
            SARTS_samples=[]
def test_PPO(agent, envs,path_):
    step1 = 0
    load_path=path_ + 'agent_weights.ckpt'
    action = np.zeros((1, num_pop))
    risk_threshold=0.01
    def smart_policy(state, action=action):
        action *= 0
        state=state[np.newaxis,:]
        risks = state[:, :, 3]
        action = np.repeat(action, state.shape[0], axis=0)
        action[risks > risk_threshold] = 1
        return  action
    s, visit = deepcopy(envs.reset())
    agent.load_weights(load_path)
    for inter in range(59):
        a,a_ = agent.pick_action(s,visit)
        s2, r, done, _,aa,visit_ = envs.step(a)
        step1 += 1
        visit = visit_
def proximal_policy_optimization_loss( advantage, old_prediction):
    loss_clipping = 0.1
    entropy_loss=0.1
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        prob=tf.reduce_sum(prob,axis=-1)
        prob=tf.reduce_mean(prob,axis=1)
        prob=tf.reshape(prob,[16,1])
        old_prob = y_true * old_prediction
        old_prob = tf.reduce_sum(old_prob, axis=-1)
        old_prob = tf.reduce_mean(old_prob, axis=1)
        old_prob = tf.reshape(old_prob, [16, 1])
        r = prob / (old_prob + 1e-10)
        confident_ratio = tf.clip_by_value(r, 1 - loss_clipping, 1 + loss_clipping)
        advantage_ = tf.reshape(advantage, [16, 1])
        current_objective = r * advantage_
        confident_objective = confident_ratio * advantage_
        PPO_objective = tf.where(current_objective < confident_objective, current_objective, confident_objective)+ entropy_loss*prob*tf.math.log(prob+1e-7)
        PPO_objective = tf.reduce_mean(PPO_objective)
        return -PPO_objective

    return loss
def get_build_func():
    def build():
        input_feature = Input(shape=(num_pop, dim_state))
        input_visit = Input(shape=(num_pop, 2,12))
        def build_actor():
            print('-' * 30, 'Building Actor', '-' * 30)
            advantage = Input(shape=(1))
            old_prediction = Input(shape=(num_pop, 4))
            feature = Dense(num_dim, activation='relu')(input_feature)
            feature = BatchNormalization()(feature)
            visit = input_visit
            for i in range(num_his):
                visit_ = Lambda(lambda x: x[:, :, i])(visit)
                feature = GNNLayer([num_dim, num_dim], BNs=[BatchNormalization(), BatchNormalization()],
                                   activation='relu', kernel_regularizer=l2(5e-4))([feature, visit_])
            action=Dense(4)(feature)
            action = Activation('softmax')(action)
            actor = Model([input_feature, advantage,input_visit,old_prediction], action)
            actor.compile(optimizer=Adam(lr=1e-4),
                          loss=proximal_policy_optimization_loss(
                              advantage=advantage, old_prediction=old_prediction,
                          ), metrics=['mae'], experimental_run_tf_function=False)
            actor.summary()

            return actor

        def build_critic():
            print('-' * 30, 'Building Critic', '-' * 30)
            feature = Dense(num_dim, activation='relu')(input_feature)
            feature = BatchNormalization()(feature)
            visit=input_visit
            for i in range(num_his):
                visit_ = Lambda(lambda x: x[:, :, i])(visit)
                feature = GNNLayer([num_dim, num_dim], BNs=[BatchNormalization(), BatchNormalization()],
                                   activation='relu', kernel_regularizer=l2(5e-4))([feature, visit_])
            reward = Lambda(lambda x: tf.reduce_mean(x, axis=1))(feature)
            reward = Dense(num_dim, activation='relu')(reward)
            reward = Dense(1)(reward)
            critic = Model([input_feature,input_visit], reward)
            critic.compile(optimizer=Adam(lr=1e-4,clipnorm=1.), loss='mean_squared_error', metrics=['mae'],
                           experimental_run_tf_function=False)
            critic.summary()
            return critic

        return build_actor, build_critic

    return build
if __name__ == '__main__':
    def get_reward_func(args):
        def reward_func(i,q,hos,conf1,quar1):
            def r1(i1, q1, hos,conf,quar):
                r = -(np.exp(i1 / args.i_threshold) + np.exp((0.2*conf+0.3*quar+0.5 * q1 + hos) / args.q_threshold))
                Q=0.2*conf+0.3*quar+0.5 * q1 + hos
                return r,Q
            end = False
            re,Q=r1(i,q,hos,conf1,quar1)
            return re, end

        return reward_func
  
    train=args.train
    
    reward_func = get_reward_func(args)
    envs = City_env(reward_func=reward_func, period=args.period, num_pop=args.num_pop, thread_num=args.thread_num,
                   fixed_no_policy_days=args.fixed_no_policy_days, name=args.save_name,Scenario=args.scenario)
    build_function = get_build_func()
    build_actor, build_critic = build_function()
    agent=DiscretePPO(build_critic,build_actor)
    if train:
        train_PPO(agent, envs,args.epochs,args.save_path)
    else:
        test_PPO(agent,envs,args.load_path)
