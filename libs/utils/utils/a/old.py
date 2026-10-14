
#class TopkHeterodyneAugmentor(torch.nn.Module):
#    def __init__(
#        self,
#        phase_file: str,
#        sample_rate: float,
#        kernel_length: float,
#        num_chirp_masses: int,
#        k: int,
#        highpass: float = 0,
#        lowpass: float = 1024,
#        keep_last_n_seconds: float = None,
#    ):
#        super().__init__()
#        self.sample_rate = sample_rate
#        self.kernel_length = kernel_length
#        self.keep_last_n_seconds = keep_last_n_seconds
#        self.phase_file = phase_file
#        self.num_chirp_masses = num_chirp_masses
#        if self.keep_last_n_seconds is not None:
#            self.keep_last_n_samples = int(
#                self.keep_last_n_seconds * sample_rate
#            )
#        
#        self.heterodyne_transform = Heterodyne(
#            sample_rate=self.sample_rate,
#            kernel_length=self.kernel_length,
#            chirp_mass_file=self.phase_file,
#            return_type="time",
#            highpass = highpass,
#            lowpass = lowpass,
#        )
#        self.k = k
#    
#    def topk_bin(self, X, _M, _B):
#        pooled = F.avg_pool1d(X.abs(), kernel_size = 31, stride = 5, padding = 0)
#        hl = torch.max(pooled[:, :_M]*pooled[:, _M:], dim = -1)[0]
#        pred = hl.topk(self.k, dim = -1)[1]
#        pred = torch.concat([pred, pred+_M], dim = -1)
#        pred = pred.unsqueeze(-1).expand(-1, -1, X.size(-1))
#        return torch.gather(X, dim=1, index=pred)
#    
#    def forward(self, x: Tensor) -> Tensor:
#        _B, _C, _T = x.shape
#        if self.keep_last_n_seconds is not None:
#            X_td = x[..., -self.keep_last_n_samples :].float().clone()
#        else:
#            X_td = x.float().clone()
#        
#        x = self.heterodyne_transform(x)
#        x = x.reshape(_B, _C * self.num_chirp_masses, _T)
#        x = self.topk_bin(x, self.num_chirp_masses, _B)
#        if self.keep_last_n_seconds is not None:
#            x = x[..., -self.keep_last_n_samples:]
#        
#        return torch.cat([X_td[:, :1, :], x[:, :self.k, :], X_td[:, 1:, :], x[:, self.k:, :]], dim=1).to(torch.float32)


#class TopkHeterodyneAugmentor(torch.nn.Module):
#    def __init__(
#        self,
#        phase_file: str,
#        sample_rate: float,
#        kernel_length: float,
#        num_chirp_masses: int,
#        k: int,
#        highpass: float = 0,
#        lowpass: float = 1024,
#        keep_last_n_seconds: float = None,
#    ):
#        super().__init__()
#        self.sample_rate = sample_rate
#        self.kernel_length = kernel_length
#        self.keep_last_n_seconds = keep_last_n_seconds
#        self.phase_file = phase_file
#        self.num_chirp_masses = num_chirp_masses
#        if self.keep_last_n_seconds is not None:
#            self.keep_last_n_samples = int(
#                self.keep_last_n_seconds * sample_rate
#            )
#        
#        self.heterodyne_transform = Heterodyne(
#            sample_rate=self.sample_rate,
#            kernel_length=self.kernel_length,
#            chirp_mass_file=self.phase_file,
#            return_type="time",
#            highpass = highpass,
#            lowpass = lowpass,
#        )
#        self.k = k
#    
#    def topk_bin(self, X, _M, _B):
#        pooled = F.avg_pool1d(X.abs(), kernel_size = 31, stride = 5, padding = 0)
#        hl = torch.max(pooled[:, :_M]*pooled[:, _M:], dim = -1)[0]
#        pred = hl.topk(self.k, dim = -1)[1]
#        pred = torch.concat([pred, pred+_M], dim = -1)
#        pred = pred.unsqueeze(-1).expand(-1, -1, X.size(-1))
#        return torch.gather(X, dim=1, index=pred), pred
#    
#    def forward(self, x: Tensor) -> Tensor:
#        _B, _C, _T = x.shape
#        if self.keep_last_n_seconds is not None:
#            X_td = x[..., -self.keep_last_n_samples :].float().clone()
#        else:
#            X_td = x.float().clone()
#        
#        x = self.heterodyne_transform(x)
#        x = x.reshape(_B, _C * self.num_chirp_masses, _T)
#        x, pred = self.topk_bin(x, self.num_chirp_masses, _B)
#        if self.keep_last_n_seconds is not None:
#            x = x[..., -self.keep_last_n_samples:]
#        
#        return torch.cat([X_td[:, :1, :], x[:, :self.k, :], X_td[:, 1:, :], x[:, self.k:, :]], dim=1).to(torch.float32), pred


#class NbhdHeterodyneAugmentor(torch.nn.Module):
#    def __init__(
#        self,
#        phase_file: str,
#        offsets: list[int],
#        sample_rate: float,
#        kernel_length: float,
#        num_chirp_masses: int,
#        highpass: float = 0,
#        lowpass: float = 1024,
#        keep_last_n_seconds: float = None,
#    ):
#        super().__init__()
#        self.sample_rate = sample_rate
#        self.kernel_length = kernel_length
#        self.register_buffer("offsets", torch.tensor(offsets), dtype=torch.float32)
#        self.keep_last_n_seconds = keep_last_n_seconds
#        self.phase_file = phase_file
#        self.num_chirp_masses = num_chirp_masses
#        if self.keep_last_n_seconds is not None:
#            self.keep_last_n_samples = int(
#                self.keep_last_n_seconds * sample_rate
#            )
#        
#        self.heterodyne_transform = Heterodyne(
#            sample_rate=self.sample_rate,
#            kernel_length=self.kernel_length,
#            chirp_mass_file=self.phase_file,
#            return_type="time",
#            highpass = highpass,
#            lowpass = lowpass,
#        )
#    
#    def nbhd_bin(self, X, _M, _B):
#        pooled = F.avg_pool1d(X.abs(), kernel_size = 31, stride = 5, padding = 0)
#        
#        hl = torch.max(pooled[:, :_M]*pooled[:, _M:], dim = -1)[0]
#        idx = torch.argmax(hl, dim = -1)
#        idx = idx.unsqueeze(1).repeat(1, 5)+self.offsets
#        idx = torch.clip(idx, min=0, max=99)
#        
#        idx = torch.concat([idx, idx+_M], dim = -1) #get both h and l channels
#        return X[torch.arange(_B).unsqueeze(-1), idx]
#    
#    def forward(self, x: Tensor) -> Tensor:
#        _B, _C, _T = x.shape
#        if self.keep_last_n_seconds is not None:
#            X_td = x[..., -self.keep_last_n_samples :].float().clone()
#        else:
#            X_td = x.float().clone()
#        
#        x = self.heterodyne_transform(x)
#        x = x.reshape(_B, _C * self.num_chirp_masses, _T)
#        x = self.nbhd_bin(x, self.num_chirp_masses, _B)
#        if self.keep_last_n_seconds is not None:
#            x = x[..., -self.keep_last_n_samples:]
#        
#        return torch.cat([X_td[:, :1, :], x[:, :len(self.offsets), :], X_td[:, 1:, :], x[:, len(self.offsets):, :]], dim=1).to(torch.float32)


#class NbhdHeterodyneAugmentor(torch.nn.Module):
#    def __init__(
#        self,
#        phase_file: str,
#        offsets: list[int],
#        sample_rate: float,
#        kernel_length: float,
#        num_chirp_masses: int,
#        highpass: float = 0,
#        lowpass: float = 1024,
#        keep_last_n_seconds: float = None,
#    ):
#        super().__init__()
#        self.sample_rate = sample_rate
#        self.kernel_length = kernel_length
#        self.register_buffer("offsets", torch.tensor(offsets))
#        self.keep_last_n_seconds = keep_last_n_seconds
#        self.phase_file = phase_file
#        self.num_chirp_masses = num_chirp_masses
#        if self.keep_last_n_seconds is not None:
#            self.keep_last_n_samples = int(
#                self.keep_last_n_seconds * sample_rate
#            )
#        
#        self.heterodyne_transform = Heterodyne(
#            sample_rate=self.sample_rate,
#            kernel_length=self.kernel_length,
#            chirp_mass_file=self.phase_file,
#            return_type="time",
#            highpass = highpass,
#            lowpass = lowpass,
#        )
#    
#    def nbhd_bin(self, X, _M, _B):
#        pooled = F.avg_pool1d(X.abs(), kernel_size = 31, stride = 5, padding = 0)
#        
#        hl = torch.max(pooled[:, :_M]*pooled[:, _M:], dim = -1)[0]
#        idx = torch.argmax(hl, dim = -1)
#        idx = idx.unsqueeze(1).repeat(1, 5)+self.offsets
#        idx = torch.clip(idx, min=0, max=99)
#        
#        idx = torch.concat([idx, idx+_M], dim = -1) #get both h and l channels
#        return X[torch.arange(_B).unsqueeze(-1), idx]
#    
#    def forward(self, x: Tensor) -> Tensor:
#        _B, _C, _T = x.shape
#        x_heterodyned = torch.empty((_B, _C * len(self.offsets), _T))
#        x = self.heterodyne_transform(x)
#        x = x.reshape(_B, _C * self.num_chirp_masses, _T)
#        x = self.nbhd_bin(x, self.num_chirp_masses, _B)
#        x_heterodyned[:, :, :] = x
#        if self.keep_last_n_seconds is not None:
#            return x_heterodyned[..., -self.keep_last_n_samples :]
#        else:
#            return x_heterodyned