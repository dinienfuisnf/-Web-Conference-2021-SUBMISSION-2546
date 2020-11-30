from gym import spaces, Env
import simulator
import numpy as np
from keras.utils import np_utils
dim_state = 9
pop_num=10000

class City_env(Env):
    def __init__(self, reward_func, num_pop=10000, thread_num=8, period=840, fixed_no_policy_days=1, contact_his=3,
                 intervene_his=14, name='test',Scenario='scenario1'):

        self.K = 1
        self.population = num_pop
        self.engine = simulator.Engine(thread_num=thread_num, write_mode="write", specified_run_name=name,scenario=Scenario)


        print('Build engine successed')

        self.action_space = spaces.MultiBinary(self.population * self.K)
        self.time_step = 0
        self.period = period
        self.state = None
        self.fixed_no_policy_days = fixed_no_policy_days
        self.reward_func = reward_func
        self.prob_s = 0.01
        self.prob_c = 0.05
        self.write=np.zeros((60,100))
        self.time=0


    def get_state(self):
        s1 = np.array([self.engine.get_individual_infection_state(i) for i in range(self.population)]) - 1
        current_intervene = np.array(
            [self.engine.get_individual_intervention_state(i) for i in range(self.population)]) - 1

        s1 = np_utils.to_categorical(s1, num_classes=5)  # change to one-hot vector
        s2 = np_utils.to_categorical(current_intervene, num_classes=5)

        s5 = self.engine.get_current_day() * np.ones((10000, 1))
        s6 = self.engine.get_acquaintance_count() * np.ones((10000, 1))
        s7 = self.engine.get_stranger_count() * np.ones((10000, 1))

        s9=np.array([len(self.engine.get_individual_residential_acq(i)) for i in range(self.population)])
        s9=s9.reshape(self.population,1)
        s9_=np.array([len(self.engine.get_individual_working_acq(i)) for i in range(self.population)])
        s9_=s9_.reshape(self.population,1)

        s3 = np.array([self.engine.get_individual_visited_history(i) for i in range(self.population)])[:,
             :14]  # pop * 70 one day's visit history
        s3_ = s3.reshape((-1, 2, 7)) + 1  # plus one is for intervention
        s3_ = np_utils.to_categorical(s3_, 12)
        s3_ = s3_.sum(-2)  # accumlate one day, pop * 2 * 12
        current_infection = np.array([self.engine.get_individual_infection_state(i) for i in range(self.population)])
        current_symptomatic = np.bitwise_or(current_infection == 3, current_infection == 4)
        current_symptomatic_set = set(np.where(current_symptomatic)[0])
        current_notsymptomatic_set = set(np.where(~current_symptomatic)[0])
        num_areas = len(self.engine.get_all_area_category())
        area_category=self.engine.get_all_area_category()
        prob_not_infected_by_discovered = np.ones(10000)  # Initialize prob_not_infected_by_discovered
        prob_not_infected_by_discovered[current_symptomatic] = 0
        for loc_id in range(num_areas):
            his = self.engine.get_area_visited_history(loc_id)
            if area_category[loc_id]==2:
                for his_ in his:
                    his_set = set(his_)
                    his_discovered = his_set & current_symptomatic_set
                    his_healthy = his_set & current_notsymptomatic_set
                    his_prob_infect = self.prob_s * len(his_discovered) / (len(his_) + 1e-7)
                    prob_not_infected_by_discovered[list(his_healthy)] *= 1 - his_prob_infect
            else:
                for his_ in his:
                    familiar = []
                    his_set = set(his_)
                    his_discovered = his_set & current_symptomatic_set
                    his_discovered_ = list(his_discovered)
                    for inter in his_discovered_:
                        familiar += self.engine.get_individual_residential_acq(inter)
                        familiar += self.engine.get_individual_working_acq(inter)
                    his_healthy = his_set & current_notsymptomatic_set
                    familiar = set(familiar)
                    familiar_ = familiar & his_healthy
                    his_healthy = his_healthy.difference(familiar_)
                    his_prob_infect = self.prob_s * len(his_discovered) / (len(his_) + 1e-7)
                    prob_not_infected_by_discovered[list(his_healthy)] *= 1 - his_prob_infect
                    prob_not_infected_by_discovered[list(familiar_)] *= 1 - self.prob_c
        pos = np.where(current_intervene == 5)[0]
        pos1 = np.where(current_infection == 5)[0]
        prob_not_infected_by_discovered[pos] = 1
        prob_not_infected_by_discovered[pos1] = 1
        p_infecious = 1 - prob_not_infected_by_discovered
        p_infecious = p_infecious.reshape(self.population, 1)
        state = np.hstack((s1[:, :1], s1[:, 2:3] + s1[:, 3:4], s1[:, 4:], p_infecious))

        return state, s3_

    def reset(self):
        self.engine.reset()
        self.engine.set_random_seed(11)
        self.time_step = 0
        self.q = np.zeros((200,1))
        self.state = None
        for i in range(self.fixed_no_policy_days*14):  # 2days no intervene days
            self.engine.next_step()
            self.time_step += 1
        self.state, self.daily_visit = self.get_state()
        return self.state,self.daily_visit

    def get_isolated(self):
        current_intervene = np.array([self.engine.get_individual_intervention_state(i) for i in range(self.population)])
        return np.sum(current_intervene == 4)

    def step(self, action):
        action = action.reshape(-1,4)
        current_intervene = np.array([self.engine.get_individual_intervention_state(i) for i in range(self.population)])
        # current_recovered = self.state[:,2]
        current_infection = np.array([self.engine.get_individual_infection_state(i) for i in range(self.population)])
        recovered = np.where(current_infection == 5)[0]
        set_treat = np.bitwise_and(current_infection == 3, current_intervene != 5)
        set_treat = np.where(set_treat)[0]
        action[set_treat] = 0
        action[recovered] = 0
        set_conf=np.where(action[:,1]==1)[0]
        set_quar=np.where(action[:,2]==1)[0]
        set_isolate = np.where(action[:,3] == 1)[0]
        set_individual_conf_days = {i: 1 for i in set_conf}
        set_individual_quar_days = {i: 1 for i in set_quar}
        set_individual_isolate_days = {i: 1 for i in set_isolate}
        set_individual_hospitalize = {i: True for i in set_treat}

        self.engine.set_individual_confine_days(set_individual_conf_days)
        self.engine.set_individual_quarantine_days(set_individual_quar_days)
        self.engine.set_individual_isolate_days(set_individual_isolate_days)  # {manID: day}
        self.engine.set_individual_to_treat(set_individual_hospitalize)

        for i in range(14):
            self.engine.next_step()
            self.time_step += 1

        state,visit = self.get_state()
        life=state[:, 0].sum()
        done = True if self.time_step == self.period else False
        i = self.state[:, 0].sum() - state[:, 0].sum()
        current_intervene_ = np.array(
            [self.engine.get_individual_intervention_state(i) for i in range(self.population)])
        current_intervene_ = np.where(current_intervene_ == 4)
        current_intervene_d = np.zeros((10000, 1))
        current_intervene_d[current_intervene_] = 1
        current_intervene_d=current_intervene_d.reshape(200,50)
        current_intervene_ = np.sum(current_intervene_d, axis=-1)
        q = current_intervene_
        current_intervene = np.array([self.engine.get_individual_intervention_state(i) for i in range(self.population)])
        current_infection = np.array([self.engine.get_individual_infection_state(i) for i in range(self.population)])
        recovered=np.bitwise_and(current_infection == 5, current_infection == 5)
        che = np.bitwise_and(current_intervene == 5, current_intervene == 5)
        conf=np.bitwise_and(current_intervene == 2, current_intervene == 2)
        quar=np.bitwise_and(current_intervene == 3, current_intervene == 3)
        conf=conf.sum()
        quar = quar.sum()
        che = che.sum()
        q=q.sum()
        reward,end_ = self.reward_func(i,q,che,conf,quar)
        self.R=R
        delta_reward = reward - self.reward_ago
        print(np.mean(reward), np.mean(che), np.mean(delta_reward))
        self.reward_ago = reward
        self.state = state
        return state, reward, done, {}, end_,visit
