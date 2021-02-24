import numpy as np
import time

class Bull():
    """描述游戏环境的类"""

    def __init__(self):
        """初始化类的属性"""
        self.policy = np.ones((6, 6, 6, 6, 6, 6, 9, 7, 9, 7)) / 37
        self.policy[:, :, :, :, :, :, :, :, :3, :] = 0
        self.policy[:, :, :, :, :, :, :, :, :, 0] = 0
        self.policy[:, :, :, :, :, :, :, :, 0, 1] = 1/37
        self.player = np.array([0, 0, 0, 0, 0, 0])
        self.dealer = np.array([0, 0, 0, 0, 0, 0])
        self.player_display = np.array([0, 0, 0, 0, 0, 0])
        self.dealer_display = np.array([0, 0, 0, 0, 0, 0])
        self.player_wtihone = np.array([0, 0, 0, 0, 0, 0])
        self.dealer_wtihone = np.array([0, 0, 0, 0, 0, 0])
        self.q = np.zeros_like(self.policy)
        self.c = np.zeros_like(self.policy)
        self.q[:,:,:,:,:,:,:,:,:3,:]=float('-inf')
        self.q[:,:,:,:,:,:,:,:,:,0]=float('-inf')
        self.q[:, :, :, :, :, :, :, :, 0, 1] = 0
        self.episode_num = 5000000000
        self.epilion = 0.1
        self.one_token = 0
        self.single_token = 0
        self.leo_token = 0

    def reset(self):
        """初始化一局游戏，返回自己摇到的筛子和对方喊的点数"""
        self.one_token = 0
        self.player = np.array([0, 0, 0, 0, 0, 0])
        self.dealer = np.array([0, 0, 0, 0, 0, 0])
        player = np.random.randint(1, 7, 5)
        dealer = np.random.randint(1, 7, 5)
        for i in range(5):
            self.player[player[i] - 1] += 1
            self.dealer[dealer[i] - 1] += 1
        self.player_display = np.copy(self.player)
        self.dealer_display = np.copy(self.dealer)
        self.player_wtihone = np.copy(self.player)
        self.dealer_wtihone = np.copy(self.dealer)
        for i in range(1, 6):
            self.player_wtihone[i] = self.player_wtihone[i] + self.player_wtihone[0]
            self.dealer_wtihone[i] = self.dealer_wtihone[i] + self.dealer_wtihone[0]
        if max(self.dealer) == 1:#单骰为0
            self.single_token = 1
            self.dealer = np.array([0, 0, 0, 0, 0, 0])
            self.dealer_wtihone = np.array([0, 0, 0, 0, 0, 0])
        else:
            self.single_token = 0
        if max(self.player) == 1:#单骰为0
            self.player = np.array([0, 0, 0, 0, 0, 0])
            self.player_wtihone = np.array([0, 0, 0, 0, 0, 0])
        if 5 in self.dealer_wtihone:
            self.leo_token = np.argmax(self.dealer_wtihone) + 1
        else:
            self.leo_token = 0

    def oberv(self):
        return self.player

    def step(self, player, dealer):
        """判断是继续喊还是直接开判断输赢"""
        if player[0]==0 and player[1] !=1 : #乱喊 输
            return (1,-1,0)
        if player[0] == 0 and player[1] == 1:  # 玩家开
            if self.one_token == 1:  # 1已经用过
                if (self.dealer[dealer[1] - 1] + self.player[dealer[1] - 1]) < dealer[0]:  # 玩家赢了
                    #print('最后观察',self.dealer,self.player)
                    return (1, 10, 0)
                else:
                    #print('最后观察', self.dealer, self.player)
                    return (1, -1, 0)
            else:  # 1没用过
                if (self.dealer_wtihone[dealer[1] - 1] + self.player_wtihone[dealer[1] - 1]) < dealer[0]:  # 玩家赢了
                    #print('最后观察', self.dealer_wtihone, self.player_wtihone)
                    return (1, 10, 0)
                else:
                    #print('最后观察', self.dealer_wtihone, self.player_wtihone)
                    return (1, -1, 0)
        else:  # 玩家不开
            if self.one_token == 1:  # 1已经用过
                de = self.call(player, self.dealer)
                if de[0] == 0 and de[1] == 1:  # 庄家开
                    if (self.dealer[player[1] - 1] + self.player[player[1] - 1]) < player[0]:  # 庄家赢了
                        return (1, -1, 0)
                    else:
                        return (1, 10, 0)
                else:  # 庄家不开
                    return (0, 0, de)
            else:  # 1没用过
                de = self.call(player, self.dealer_wtihone)
                if de[0] == 0 and de[1] == 1:  # 庄家开
                    if (self.dealer_wtihone[player[1] - 1] + self.player_wtihone[player[1] - 1]) < player[0]:  # 庄家赢了
                        return (1, -1, 0)
                    else:
                        return (1, 10, 0)
                else:
                    return (0, 0, de)

    def call(self, player, dealer):
        """庄家根据玩家喊的喊，1喊过则置标志位为1"""
        if M==0:
            if player[0] == 10:  # 庄家先喊
                if self.single_token:  # 单骰
                    return np.array([3, 6])
                elif self.leo_token != 0 and self.one_token == 0 or (
                        self.leo_token != 0 and self.one_token == 1 and self.player[self.leo_token - 1] == 5):  # 豹子
                    if self.leo_token == 1:
                        self.one_token = 1
                    return np.array([3, self.leo_token])
                else:  # 非特殊情况
                    if np.random.rand()<0.00001: #乱喊一通
                        a = np.random.randint(3,9)
                        b = np.random.randint(1,7)
                        return np.array([a,b])
                    prob_dealer = np.copy(dealer)
                    prob_dealer = np.exp(prob_dealer)
                    prob_dealer = prob_dealer / sum(prob_dealer)
                    n = np.random.choice([1, 2, 3, 4, 5, 6], p = prob_dealer)
                    if n == 1:
                        self.one_token = 1
                    if np.random.rand() < 0.6:
                        return np.array([3, n])
                    else:
                        return np.array([4, n])

            else:  # 玩家先喊
                if self.single_token:  # 单骰
                    if player[0] <= 3 and player[1] < 6:
                        return np.array([3, 6])
                    else:
                        return np.array([0, 1])
                elif self.leo_token != 0 and self.one_token == 0 or (
                        self.leo_token != 0 and self.one_token == 1 and self.player[self.leo_token - 1] == 5):  # 豹子
                    if self.leo_token == 1:
                        self.one_token = 1
                    if self.leo_token > player[1]:  # 豹子点数比玩家点数大
                        return np.array([player[0], self.leo_token])
                    else:  # 个数加一
                        if player[0] == 8:  # 个数点数都喊不上去了，直接开
                            return np.array([0, 1])
                        else:
                            return np.array([player[0] + 1, self.leo_token])
                else:  # 非特殊情况
                    # 计算对面的点数数量的概率
                    if abs(player[0]-dealer[player[1]-1]) >2:
                        if np.random.rand() < 0.7:
                            return np.array([0,1])
                    temp = player[0] - 1
                    prob = np.zeros(7)
                    for i in range(7):
                        prob[i] = max(0, temp - abs(temp - i))
                    prob = np.exp(prob)
                    prob = prob / sum(prob)
                    point_num = np.random.choice(np.array([0, 1, 2, 3, 4, 5, 6]), p = prob)  # point_num为估计对方拥有所喊点数的个数
                    temp_point = np.max(np.where(dealer == np.max(dealer))) + 1 #计算自己那个骰子最多
                    temp_num = dealer[temp_point - 1] #自己最多的骰子有几个
                    if point_num + dealer[player[1] - 1] < player[0]:  # 如果认为场上骰子数没有玩家喊的多
                        return np.array([0, 1])  # 开
                    elif point_num + dealer[player[1] - 1] > player[0]:  # 如果认为场上骰子数比玩家喊得多
                        if np.random.rand()>0.2:
                            if player[0] == 8:  # 个数点数都喊不上去了，直接开
                                return np.array([0, 1])
                            else:
                                return np.array([player[0] + 1, player[1]])  # 0.8的概率加喊一个
                        else:
                            if np.random.rand() > 0.3: #不加一个 0.7概率直接喊最大的
                                return np.array([min(point_num + dealer[player[1] - 1],8), player[1]])  # 0.7的概率直接喊最大值
                            else: #0.4概率自己喊
                                if temp_point == 1:
                                    self.one_token = 1
                                if temp_point > player[1]:
                                    return np.array([player[0], temp_point])
                                else:
                                    return np.array([min(player[0] + 1,8), temp_point])


                    else:  # 如果认为场上骰子数和玩家喊得一样多
                        if np.random.rand() < 0.1:  # 0.1概率加一个
                            if player[0] == 8:  # 个数点数都喊不上去了，直接开
                                return np.array([0, 1])
                            else:
                                  return np.array([player[0] + 1, player[1]])
                        else:
                            if temp_point == 1:
                                self.one_token = 1
                            if temp_num < point_num:  # 如果自己最多的骰子数比估计出玩家有的那颗骰子数少
                                return np.array([0, 1])  # 开
                            else:
                                if temp_point > player[1]:
                                    return np.array([player[0], temp_point])
                                else:
                                    return np.array([min(player[0] + 1,8), temp_point])
        else: #对战模式
            de = list(map(int, input("输入个数点数").strip().split()))
            if de[1]==1:
                bull.one_token =1
            return de

bull = Bull()
bull.policy=np.load('C:/Users/Dell/Desktop/policy.npy')
count = 0
train_count = 0
win = 0
acc=0
save_token=0
M=int(input("训练:0，对战:1"))
start = time.time()
while acc<0.9:
    if M == 1:
        train_count += 1
        if train_count % 5 == 0:
            np.save('C:/Users/Dell/Desktop/policy.npy', bull.policy)
    end=time.time()
    if end-start>600: #10分钟保存一次
        np.save('C:/Users/Dell/Desktop/policy.npy', bull.policy)
        start=time.time()
    count=count+1
    if count%10000 == 0:
        acc=win/10000
        print('acc',acc)
        #print(bull.policy[state][a])
        #print(state)
        #print(a)

        win=0
        count=0
    reward = 0;
    state_action = []
    bull.reset()
    bull.one_token=0
    observ_player = bull.oberv()
    print("你的骰子")
    print(bull.dealer_display)
    #print("AI的骰子")
    #print(bull.player_display)
    if np.random.rand() < 0.5:  # 玩家先喊
        print("AI先喊")
        de = np.array([0, 0])
        while True:
            state = np.append(observ_player, de)
            state = tuple(state.tolist())
            temp_ac = np.copy(bull.policy[state])
            temp_ac = temp_ac.reshape(63)
            #temp_ac[83] = 1 - sum(temp_ac[0:83])
            #print(temp_ac[83])
            b = np.random.choice(list(range(63)), p = temp_ac)
            a = int(b / 7)
            b = b % 7  # 动作为喊a个b
            action = (a, b)
            print("AI个数点数")
            print(action)
            state_action.append((state, action))
            if b == 1 and a !=0 :
                bull.one_token=1
            if (a < de[0] or (a == de[0] and b <= de[1])) and a != 0:  # 玩家不能喊得比庄家小,否则直接判输
                reward=-1
                print("AI犯规，你赢了")
                break
            re = bull.step(action, de)
            if re[0]:
                reward = re[1]
                if reward == 10:
                    win = win + 1
                    print('你输了')
                    print('AI的骰子')
                    print(bull.player_display)
                if reward == -1:
                    print('你赢了')
                    print('AI的骰子')
                    print(bull.player_display)
                break
            de = re[2]
        for state, action in state_action:
            bull.c[state][action] += 1
            if bull.q[state][action] != float('-inf'):
                bull.q[state][action] += (reward - bull.q[state][action]) / bull.c[state][action]
            a = bull.q[state].argmax()
            row = int(a / 7)
            line = a % 7
            a = (row, line)
            num = 37
            bull.policy[state] = bull.epilion / num
            bull.policy[state][:3, :] = 0
            bull.policy[state][3:, 0] = 0
            bull.policy[state][0, 1] = bull.epilion / num
            bull.policy[state][a] += (1 - bull.epilion)
            b = 1 - sum(sum(bull.policy[state]))
            bull.policy[state][a] = bull.policy[state][a] + b

    else:  # 庄家先喊
        print("你先喊")
        action = (10, 7)
        de = (0, 0)
        while True:
            re = bull.step(action, de)
            if re[0]:
                reward = re[1]
                if reward == 10:
                    win += 1
                    print('你输了')
                    print('AI的骰子')
                    print(bull.player_display)
                if reward == -1:
                    print('你赢了')
                    print('AI的骰子')
                    print(bull.player_display)
                break
            de = re[2]
            state = np.append(observ_player, de)
            state = tuple(state.tolist())
            temp_ac = np.copy(bull.policy[state])
            temp_ac = temp_ac.reshape(63)
            #temp_ac[83] = 1 - sum(temp_ac[0:83])
            #print(temp_ac[83])
            b = np.random.choice(list(range(63)), p = temp_ac)
            a = int(b / 7)
            b = b % 7  # 动作为喊a个b
            action = (a, b)
            print("AI个数点数")
            print(action)
            state_action.append((state, action))
            if b == 1 and a !=0:
                bull.one_token=1
            if (a < de[0] or (a == de[0] and b <= de[1])) and a != 0 :  # 玩家不能喊得比庄家小,否则直接判输
                reward=-1
                print("AI犯规，你赢了")
                break
        for state, action in state_action:
            bull.c[state][action] += 1
            if bull.q[state][action] != float('-inf'):
                bull.q[state][action] += (reward - bull.q[state][action]) / bull.c[state][action]
            a = bull.q[state].argmax()
            row = int(a / 7)
            line = a % 7
            #while line == 0:
                #print('ka3')
                #a = bull.q[state].reshape(63)[state[6] * 7 + state[7] + 1:].argmax() + state[6] * 7 + state[7]
                #row = int(a / 7)
                #line = a % 7
            a = (row, line)
            num=37
            bull.policy[state] = bull.epilion / num
            bull.policy[state][:3, :] = 0
            bull.policy[state][3:, 0] =0
            bull.policy[state][0, 1] = bull.epilion / num
            bull.policy[state][a] += (1 - bull.epilion)
            b = 1 - sum(sum(bull.policy[state]))
            bull.policy[state][a] = bull.policy[state][a] + b
    over=int(input("是否结束 Y/1 N/0"))
    if over == 1:
        break


np.save('C:/Users/Dell/Desktop/policy.npy',bull.policy)





