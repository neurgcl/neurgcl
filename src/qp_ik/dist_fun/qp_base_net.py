from abc import abstractmethod


class QPBaseNet:
    @abstractmethod
    def cal_Gamma_g_all(self, q, pts, T_w2links=None, keep_dim=False, cuda_fmt=False):
        """
        Args:
            q: (n_joint,)
            pts: (n_pts, 3)
            T_w2links: (n_link, 4, 4)
        """
        pass
    
    @abstractmethod
    def cal_Gamma_g_min(self, q, pts, T_w2links=None):
        pass

    def cal_Gamma_all(self, q, pts, T_w2links=None, keep_dim=False, cuda_fmt=False):
        """
        Args:
            q: (n_joint,)
            pts: (n_pts, 3)
            T_w2links: (n_link, 4, 4)
        """
        Gamma, _ = self.cal_Gamma_g_all(q, pts, T_w2links, keep_dim=keep_dim, cuda_fmt=cuda_fmt)
        return Gamma