class AverageMetric(object):
    def __init__(self):
        self.cnt_list = []
        self.val_list = []
    def get_avg(self):
        sum_cnt = 0
        sum_val = 0
        for val, cnt in zip(self.val_list, self.cnt_list):
            sum_val += val*cnt
            sum_cnt += cnt
        return sum_val / sum_cnt
    def reset(self):
        self.cnt_list = []
        self.val_list = []
    def updata(self, cnt, val):
        self.cnt_list.append(cnt)
        self.val_list.append(val)
