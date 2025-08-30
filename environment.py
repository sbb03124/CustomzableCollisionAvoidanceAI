import pyglet
import numpy as np
from datetime import datetime
import gym
import gym.spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
_DEBUG = __name__=='__main__'
# import other files ##########################################
import ship
###############################################################

##########################################################
# 入力の設定
# 出力の-1~1に変換するための係数
DX_DY_NORM = 20
DIST_NORM = 20*np.sqrt(2)
REL_U_NORM = 10*1.852/3600*2
##########################################################
# 報酬
SAFE_PASSING_RANGE = 2.000
##########################################################
# 環境の設定
STEP_DT = 5
EPISODE_TIME = 1800

COLLIDE_RANGE = 0.34

GOAL_RANGE = 0.500
SPACE_X_MIN = -6 # km
SPACE_X_MAX =  6 # km
SPACE_Y_MIN = -1 # km
SPACE_Y_MAX = 11 # km
SPACE_CENTER = (0, 5)

##########################################################
# 他船生成の条件
MAX_OTH_SHIP_NUM = 6
# COLLISION_TIME_MIN = 1000
COLLISION_TIME_MIN =  500
COLLISION_TIME_MAX = 1500
COG_NOISE_STD = 10
XY_NOISE_STD = 0.5
##########################################################
RENDER_WIDTH = 800
RENDER_HEIGHT = 800
##########################################################
def softsign(x):
    return x/(abs(x)+1)
##########################################################
class environment(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
        }
    def __init__(self):
        self.fix_size_observation = False        
        self.exe_time = datetime.now().strftime('%Y%m%d%H%M%S')
        self.episode_count = 0
        self.test_mode = False
        # action space & observation space
        self.action_space = gym.spaces.Box(low = -np.ones((1,)), high = np.ones((1,)))
        self.observation_space = gym.spaces.Box(low = -np.ones(3+10*MAX_OTH_SHIP_NUM), high = np.ones(3+10*MAX_OTH_SHIP_NUM))
        # Env の設定
        self.step_dt = STEP_DT
        self.oth_num = 0
        # Agentの設定
        self.OwnShip = ship.KTmodel()
        self.OtherShips = [ ship.KTmodel() for _ in range(MAX_OTH_SHIP_NUM)]
        self.Dist2Oth = [None for _ in range(MAX_OTH_SHIP_NUM)]
        # renderの設定
        self.viewer = None
        self.render_height = RENDER_WIDTH
        self.render_width = RENDER_HEIGHT
        self.render_scale = min(
            [
                RENDER_HEIGHT/(SPACE_Y_MAX-SPACE_Y_MIN),
                RENDER_WIDTH/(SPACE_X_MAX-SPACE_X_MIN)
            ]
        )*0.8
        self.render_center = ( self.render_height/2, self.render_width/2 )


        self.evaluation = [None for _ in range(MAX_OTH_SHIP_NUM+1)]

    def _seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.episode_count += 1
        if self.test_mode:
            np.random.seed(self.episode_count)
        self.sim_time = 0
        # reset own ship
        self.OwnShip.reset( x_init=0, y_init=0, sog_init=10*1.852/3600, head_init=0, rot_init=0 )
        self.own_bumper = np.random.uniform(
            low  = self.OwnShip._Loa*1,
            high = self.OwnShip._Loa*8,
            size=(4,)
        )
        # reset oth ship
        ## 他船の数をランダムに決定
        self.oth_num = np.random.randint(MAX_OTH_SHIP_NUM+1)
        self._reset_others(self.oth_num)
        # reset goal
        # goalの位置は自船前方に設定
        self.OwnShip._Heading = np.random.uniform(-np.pi/6,np.pi/6)
        goal_dir = 0
        dist2goal = 10*1.852/3600 * EPISODE_TIME * 5
        self._goal = np.array(
            [
                dist2goal*np.sin(goal_dir),
                dist2goal*np.cos(goal_dir),   
            ]
        )

        RelGoal = ( self._goal - np.array(self.OwnShip.data[:2]) )@np.array([[np.cos(np.radians(self.OwnShip.data[3])),np.sin(np.radians(self.OwnShip.data[3]))],[-np.sin(np.radians(self.OwnShip.data[3])),np.cos(np.radians(self.OwnShip.data[3]))]])
        self.dir2goal = np.degrees(np.arctan2(*RelGoal))%360
        if self.dir2goal>180:
            self.dir2goal -= 360

        self.evaluation = [[None for _ in range(22)] for _ in range(self.oth_num+1)]
        self.attention = [None for _ in range(self.oth_num)]
        self.rewards_for_log = [None for _ in range(self.oth_num+2)]
        if self.test_mode: self._log()
        return self._cal_state()

    def _reset_others(self, other_num):
        """
        他船の初期化用の関数
        """
        for idx in range(other_num):
            while True:
                init_cog = ( self.OwnShip.data[3] + np.random.uniform(low=30, high=330) )%360
                init_sog = ( 8 + np.random.random()*4 )*1.852/3600

                # 衝突までの時間を決定
                t = np.random.uniform( high=COLLISION_TIME_MAX, low=COLLISION_TIME_MIN )
                # ランダムに決定した衝突時間から初期位置を計算
                init_x = self.OwnShip.data[0] + (self.OwnShip.data[2]*np.sin(np.radians(self.OwnShip.data[3])) - init_sog*np.sin(np.radians(init_cog)) )*t
                init_y = self.OwnShip.data[1] + (self.OwnShip.data[2]*np.cos(np.radians(self.OwnShip.data[3])) - init_sog*np.cos(np.radians(init_cog)) )*t
                if not self.test_mode:
                    init_x   += np.random.normal(scale=XY_NOISE_STD)
                    init_y   += np.random.normal(scale=XY_NOISE_STD)
                    init_cog += np.random.normal(scale=COG_NOISE_STD)
                    
                init_rot = 0
                self.OtherShips[idx].reset( init_x, init_y, init_sog, init_cog, init_rot )
                
                # # 相手船の初期位置がバンパー外にいるか確認
                if (init_x-self.OwnShip._X)**2 + (init_y-self.OwnShip._Y)**2 >= (8*0.34)**2:
                    break

            rel_posi = (
                np.array(self.OtherShips[idx].data[:2])-np.array(self.OwnShip.data[:2])
            )@np.array(
                [
                    [np.cos(np.radians(self.OwnShip.data[3])),np.sin(np.radians(self.OwnShip.data[3]))],
                    [-np.sin(np.radians(self.OwnShip.data[3])),np.cos(np.radians(self.OwnShip.data[3]))]
                ]
            )
            self.Dist2Oth[idx] = np.sqrt(
                (
                    rel_posi[0]/1.6/0.34 if rel_posi[0]>=0 else rel_posi[0]/1.6/0.34
                )**2
                + (
                    rel_posi[1]/6.4/0.34 if rel_posi[1]>=0 else rel_posi[1]/1.6/0.34
                )**2
            )


        for idx in range(other_num, MAX_OTH_SHIP_NUM):
            self.OtherShips[idx].reset(
                -1000, -1000, 0, 0, 0
            )
            self.Dist2Oth[idx]=None
        
    def step(self, action):
        self.sim_time += self.step_dt
        done = False
        # 自船の動きを計算
        ## 舵角を代入
        rudder = action[0]*5
        ## 操舵量を計算
        rudder_change = rudder - self.OwnShip._rudder
        ## 自船を1step進める
        self.OwnShip.step(rudder, STEP_DT)
        ## 他船の動きを計算
        [ oth.step(0, STEP_DT) for oth in self.OtherShips]
        
        # 報酬を計算
        reward = self._cal_reward()
        # 衝突の判定
        done, r = self._judge_done()   
        ##############################################################################################
        if self.test_mode: self._log()
        ##############################################################################################
        return self._cal_state(), reward, done, {}
    
    def _cal_state(self):
        """
        AIの観測する状態を計算する関数
        現状の観測は以下の通り
        """
        # 自船情報
        OwnPosi = np.array(self.OwnShip.data[:2])
        own_cog = np.radians(self.OwnShip.data[3])
        # goalに関する情報
        RelGoal = (self._goal-OwnPosi)@np.array([[np.cos(own_cog),np.sin(own_cog)],[-np.sin(own_cog),np.cos(own_cog)]])
        Goal_Dir = np.degrees(np.arctan2(*RelGoal))%360
        if Goal_Dir>180:
            Goal_Dir -= 360
        state_wp = [
            # 目的地に関する情報 dx, dy, dcog
            RelGoal[0]/DX_DY_NORM, RelGoal[1]/DX_DY_NORM, Goal_Dir/180, 
        ]
        # 他船に関する情報
        state_oth = []
        for idx, oth in enumerate(self.OtherShips):
            if idx<self.oth_num:
                OthPosi = np.array(oth.data[:2])
                RelOth  = (OthPosi-OwnPosi)@np.array([[np.cos(own_cog),np.sin(own_cog)],[-np.sin(own_cog),np.cos(own_cog)]])
                RelOth_norm = np.array(
                    [
                        (RelOth[0]/self.own_bumper[0] if RelOth[0]>=0 else RelOth[0]/self.own_bumper[2] ),
                        (RelOth[1]/self.own_bumper[1] if RelOth[1]>=0 else RelOth[1]/self.own_bumper[3] ),
                    ]
                )
                Oth_norm_Dir = np.degrees(np.arctan2(*RelOth_norm))

                RelVx = oth.data[2]*np.sin(np.radians(oth.data[3]-self.OwnShip.data[3])) - 0
                RelVy = oth.data[2]*np.cos(np.radians(oth.data[3]-self.OwnShip.data[3])) - self.OwnShip.data[2]

                u_norm = 10*1.852/3600
                RelVx_norm =  (RelVx/u_norm/self.own_bumper[0]*self.OwnShip._Loa if RelOth[0]>=0 else RelVx/u_norm/self.own_bumper[2]*self.OwnShip._Loa )
                RelVy_norm =  (RelVy/u_norm/self.own_bumper[1]*self.OwnShip._Loa if RelOth[1]>=0 else RelVy/u_norm/self.own_bumper[3]*self.OwnShip._Loa )

                # Closest Point for Approach
                dcpa = abs(np.array([-RelVy,RelVx])@(-RelOth)/np.sqrt(RelVx**2+RelVy**2+1e-12))
                tcpa = np.array([RelVx,RelVy])@(-RelOth)/(RelVx**2+RelVy**2+1e-12)
                cpa = RelOth + oth.data[2]*np.array([ RelVx, RelVy ] )*tcpa

                cpa_norm = [
                    (cpa[0]/self.own_bumper[0] if cpa[0]>=0 else cpa[0]/self.own_bumper[2] ),
                    (cpa[1]/self.own_bumper[1] if cpa[1]>=0 else cpa[1]/self.own_bumper[3] ),
                ]

                dcpa_norm = np.sqrt( cpa_norm[0]**2+cpa_norm[1]**2 )
                
                state_oth.append(
                    [
                        RelOth_norm[0]/10,
                        RelOth_norm[1]/10,
                        Oth_norm_Dir/180,
                        RelVx_norm,
                        RelVy_norm,
                        np.sqrt( RelOth_norm[0]**2 + RelOth_norm[1]**2 )/10,
                        cpa_norm[0],
                        cpa_norm[1],
                        dcpa_norm,
                        tcpa/(100*STEP_DT),
                    ]
                )
            elif self.fix_size_observation:
                # 入力の数を一定とするために，1埋めのダミーデータを代入
                state_oth.append(
                    [
                        # 相対的な運動情報, relative x, relative y, relative cog, relative vx, relative vy, relative v
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                    ]
                )


        # return np.clip(state, -1, 1)
        return np.array(state_wp), np.array(state_oth)

    def _cal_reward(self):
        self.rewards_for_log = []
        # 報酬を計算する関数
        OwnPosi = np.array(self.OwnShip.data[:2])
        own_cog = np.radians(self.OwnShip.data[3])
        ## 現状は他船との距離に応じて負の報酬(ペナルティ)を与える
        reward_col = 0
        for n in range(self.oth_num):
            # 他船との距離に応じた報酬を計算
            OthPosi = np.array(self.OtherShips[n].data[:2])
            rel_posi = (OthPosi-OwnPosi)@np.array([[np.cos(own_cog),np.sin(own_cog)],[-np.sin(own_cog),np.cos(own_cog)]])

            dist = np.sqrt(
                (
                    rel_posi[0]/self.own_bumper[0] if rel_posi[0]>=0 else rel_posi[0]/self.own_bumper[2]
                )**2
                + (
                    rel_posi[1]/self.own_bumper[1]  if rel_posi[1]>=0 else rel_posi[1]/self.own_bumper[3]
                )**2
            )
            if dist<1:
                cr = 1 - dist
                if self.Dist2Oth[n] >= dist:
                    _r = -0.5*cr - 0.20
                else:
                    _r = -1.0*cr - 0.50
                reward_col += _r
                self.rewards_for_log.append(_r)
            else:
                self.rewards_for_log.append(0)
            
            self.Dist2Oth[n] = dist

        reward_wp=0
        RelGoal = (self._goal-OwnPosi)@np.array([[np.cos(own_cog),np.sin(own_cog)],[-np.sin(own_cog),np.cos(own_cog)]])
        Goal_Dir = np.degrees(np.arctan2(*RelGoal))%360
        if Goal_Dir>180:
            Goal_Dir -= 360
        
        reward_wp = (
            0.8*np.exp(-(Goal_Dir/60)**2/2)
            + 0.2*(np.clip(1-abs(Goal_Dir)/180,0,1) )
        )
        if reward_col==0:
            _del = np.clip((abs(self.dir2goal)-abs(Goal_Dir))/min([10,abs(self.dir2goal)+0.01]), 0, 1)
            reward_wp += 0.2*_del
        
            
        self.dir2goal = Goal_Dir
        
        self.rewards_for_log.insert(0,reward_wp)
        return reward_wp, reward_col

    def _judge_done(self):
        done = False
        reward = 0
        # 自船位置[x,y]
        OwnPosi = np.array(self.OwnShip.data[:2])
        # goalに到達したかどうか. 到達したら+10の報酬
        if (self._goal - OwnPosi)@(self._goal - OwnPosi) <GOAL_RANGE**2:
            done = True
            reward += 10
        # # 空間外に出たかどうか
        # if not ((SPACE_X_MAX>OwnPosi[0]>SPACE_X_MIN) and (SPACE_Y_MAX>OwnPosi[1]>SPACE_Y_MIN)):
        #     done = True
        #     reward += -10

        # 衝突したかどうか. 衝突した時は-10の報酬
        if self._judge_collision():
            # done = True
            reward += -2
        
        # 時間切れかどうか
        if self.sim_time >= EPISODE_TIME:
            done = True
        
        return done, reward
    
    def _judge_collision(self):
        collide = False
        # 自船と他船の距離がCOLLIDE_RANGE以下なら衝突
        for idx in range(self.oth_num):
            rel =  np.array(self.OwnShip.data[:2]) - np.array(self.OtherShips[idx].data[:2])
            collide = collide or (rel@rel<COLLIDE_RANGE**2)
        return collide

    def render(self, mode = 'human', close = False):
        """
        描画用の関数
        とりあえずは重要ではないので後回しで
        """
        raise NotImplementedError
    
    def _log(self):
        import csv
        import os
        if self.sim_time==0:
            os.makedirs('test'+self.exe_time, exist_ok=True)
            with open('test'+self.exe_time+f'/test_log_{self.episode_count:02d}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    ['time'] + [ 'own'+l for l in ['_x','_y','_sog', '_hdg', '_rot', '_rudder']]
                    + sum(
                        [
                            [ f'oth{n:02d}'+l for l in ['_x','_y','_sog', '_hdg', '_rot']]
                            for n in range(self.oth_num)
                        ], []
                    ) + ['goal_x', 'goal_y']
                )
        with open('test'+self.exe_time+f'/test_log_{self.episode_count:02d}.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [self.sim_time] + self.OwnShip.data + [self.OwnShip._rudder] + sum(
                    [
                        self.OtherShips[n].data for n in range(self.oth_num)
                    ],
                    []
                ) + list(self._goal)
            )

if __name__=='__main__':
    env = environment()

    env.reset()
    env.render()
    done = False
    while not done:
        try:
            # action = int(input('Plese choose action (0~{})'.format(env.action_space.n-1)))
            action = env.action_space.sample()
        except Exception as e:
            print( 'Get Action in func(input) : {}'.format(str(e)) )
            continue
        
        a, r, done, _ =  env.step(action)
        env.set_evaluation([ n for n in np.random.random(len(env.evaluation)) ])
        env.render()
    import time
    time.sleep(1)
    env.render(close = True)