import torch
# from hydroDL.model.rnn import UH_gamma, UH_conv
from hydrodl2.core.calc import uh_conv as UH_conv, uh_gamma as UH_gamma ## zhennan revised 

"""
HBV1.1p model (Song et al., 2025) with optional state initialization.

Key additions:
- instate (bool): if True, initialize from provided init_states
- init_states: tuple (SNOWPACK, MELTWATER, SM, SUZ, SLZ), each [Ngrid, mu]
- device-safe (no hard-coded .cuda())
"""

import torch
from hydrodl2.core.calc import uh_conv as UH_conv, uh_gamma as UH_gamma


class HBV(torch.nn.Module):
    def __init__(self):
        super(HBV, self).__init__()

    def forward(
        self,
        x, parameters, staind, tdlst, mu, muwts, rtwts,
        bufftime=0, outstate=False,
        instate=False, init_states=None,
        routOpt=False, comprout=False, dydrop=False
    ):
        device = x.device
        PRECS = 1e-5

        # ----- Initialization -----
        if instate:
            assert init_states is not None and len(init_states) == 5, \
                "init_states must be a 5-tuple (SNOWPACK, MELTWATER, SM, SUZ, SLZ)"
            SNOWPACK, MELTWATER, SM, SUZ, SLZ = [t.to(device) for t in init_states]
            # Clamp to avoid negatives
            SM  = torch.clamp(SM,  min=PRECS)
            SUZ = torch.clamp(SUZ, min=PRECS)
            SLZ = torch.clamp(SLZ, min=PRECS)
            start_t = 0
        elif bufftime > 0:
            with torch.no_grad():
                xinit = x[0:bufftime, :, :]
                initmodel = HBVMulET_1_1p()
                buffpara = parameters[bufftime - 1, :, :, :]
                Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = initmodel(
                    xinit, buffpara, mu, muwts, rtwts,
                    bufftime=0, outstate=True, routOpt=False, comprout=False
                )
            start_t = bufftime
        else:
            B = x.shape[1]
            SNOWPACK = (torch.zeros([B, mu], dtype=torch.float32, device=device) + 0.001)
            MELTWATER = (torch.zeros([B, mu], dtype=torch.float32, device=device) + 0.001)
            SM        = (torch.zeros([B, mu], dtype=torch.float32, device=device) + 0.001)
            SUZ       = (torch.zeros([B, mu], dtype=torch.float32, device=device) + 0.001)
            SLZ       = (torch.zeros([B, mu], dtype=torch.float32, device=device) + 0.001)
            start_t = 0

        # Slice time after warm-up
        P     = x[start_t:, :, 0]
        T     = x[start_t:, :, 1]
        ETpot = x[start_t:, :, 2]
        Pm   = P.unsqueeze(2).repeat(1, 1, mu)
        Tm   = T.unsqueeze(2).repeat(1, 1, mu)
        ETpm = ETpot.unsqueeze(2).repeat(1, 1, mu)

        # Clamp parameters before scaling
        parAll = torch.clamp(parameters[start_t:, :, :, :], 0.0, 1.0)
        parAllTrans = torch.zeros_like(parAll)

        hbvscaLst = [
            [1, 6], [50, 1000], [0.05, 0.9], [0.01, 0.5], [0.001, 0.2],
            [0.2, 1], [0, 10], [0, 100], [-2.5, 2.5], [0.5, 10],
            [0, 0.1], [0, 0.2], [0.3, 5], [0, 1],
        ]
        routscaLst = [[0, 2.9], [0, 6.5]]

        for ip, (lo, hi) in enumerate(hbvscaLst):
            parAllTrans[:, :, ip, :] = lo + parAll[:, :, ip, :] * (hi - lo)

        Nstep, Ngrid = P.size()

        # Static vs dynamic param blend with dropout
        parstaFull = parAllTrans[staind, :, :, :].unsqueeze(0).repeat([Nstep, 1, 1, 1])
        parhbvFull = torch.clone(parstaFull)
        pmat = torch.ones([1, Ngrid, 1], device=device) * dydrop
        for ix in tdlst:
            staPar = parstaFull[:, :, ix - 1, :]
            dynPar = parAllTrans[:, :, ix - 1, :]
            drmask = torch.bernoulli(pmat).detach_().to(device)
            comPar = dynPar * (1 - drmask) + staPar * drmask
            parhbvFull[:, :, ix - 1, :] = comPar

        # Allocate outputs
        Qsimmu  = (torch.zeros(Pm.size(), dtype=torch.float32, device=device) + 0.001)
        ETmu    = (torch.zeros(Pm.size(), dtype=torch.float32, device=device) + 0.001)

        for t in range(Nstep):
            (parBETA, parFC, parK0, parK1, parK2, parLP, parPERC,
             parUZL, parTT, parCFMAX, parCFR, parCWH, parBETAET, parC) = [
                parhbvFull[t, :, ip, :] for ip in range(len(hbvscaLst))
            ]

            PRECIP = Pm[t, :, :]
            RAIN = torch.mul(PRECIP, (Tm[t, :, :] >= parTT).type(torch.float32))
            SNOW = torch.mul(PRECIP, (Tm[t, :, :] < parTT).type(torch.float32))

            # Snow
            SNOWPACK = SNOWPACK + SNOW
            melt = torch.clamp(parCFMAX * (Tm[t, :, :] - parTT), min=0.0)
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK  = SNOWPACK - melt
            refreezing = torch.clamp(parCFR * parCFMAX * (parTT - Tm[t, :, :]), min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK   = SNOWPACK + refreezing
            MELTWATER  = MELTWATER - refreezing
            tosoil = torch.clamp(MELTWATER - (parCWH * SNOWPACK), min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil & ET
            soil_wetness = torch.clamp((SM / parFC) ** parBETA, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness
            SM = SM + RAIN + tosoil - recharge
            excess = torch.clamp(SM - parFC, min=0.0)
            SM = SM - excess
            evapfactor = torch.clamp((SM / (parLP * parFC)) ** parBETAET, min=0.0, max=1.0)
            ETact = torch.min(SM, ETpm[t, :, :] * evapfactor)
            SM = torch.clamp(SM - ETact, min=PRECS)

            # Clamp state to avoid negatives
            SM  = torch.clamp(SM,  min=PRECS)
            SUZ = torch.clamp(SUZ, min=PRECS)
            SLZ = torch.clamp(SLZ, min=PRECS)

            # Groundwater
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, parPERC)
            SUZ  = SUZ - PERC
            Q0 = parK0 * torch.clamp(SUZ - parUZL, min=0.0);  SUZ = SUZ - Q0
            Q1 = parK1 * SUZ;                                  SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = parK2 * SLZ;                                  SLZ = SLZ - Q2

            Qsimmu[t, :, :]  = Q0 + Q1 + Q2
            ETmu[t, :, :]    = ETact

        Qsimave = Qsimmu.mean(-1) if muwts is None else (Qsimmu * muwts).sum(-1)

        if routOpt:
            if comprout:
                Qsim = Qsimmu.view(Nstep, Ngrid * mu)
            else:
                Qsim = Qsimave
            if rtwts is None:
                raise ValueError("routOpt=True requires rtwts with shape [B,2].")
            rtwts = torch.clamp(rtwts, 0.0, 1.0)  # clamp routing weights too
            tempa = routscaLst[0][0] + rtwts[:, 0] * (routscaLst[0][1] - routscaLst[0][0])
            tempb = routscaLst[1][0] + rtwts[:, 1] * (routscaLst[1][1] - routscaLst[1][0])
            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            UH = UH_gamma(routa, routb, lenF=15)
            rf = torch.unsqueeze(Qsim, -1).permute(1, 2, 0)
            UH = UH.permute(1, 2, 0)
            Qsrout = UH_conv(rf, UH).permute(2, 0, 1)
            Qs = Qsrout
        else:
            Qs = torch.unsqueeze(Qsimave, -1)

        return (Qs, SNOWPACK, MELTWATER, SM, SUZ, SLZ) if outstate else Qs

class HBVMulET_1_1p(torch.nn.Module):
    """
    Helper single-call HBV with optional warm-up or explicit state init.
    Used for warm-up and for cases where parameters are time-constant across window.
    """

    def __init__(self):
        super(HBVMulET_1_1p, self).__init__()

    def forward(
        self,
        x,                 # [T, B, 3]
        parameters,        # [B, n_par, mu] or [n_par, mu] broadcastable
        mu,
        muwts,
        rtwts,
        bufftime=0,
        outstate=False,
        routOpt=False,
        comprout=False,
        instate=False,
        init_states=None
    ):
        device = x.device
        PRECS = 1e-5

        # Normalize parameters shape -> [B, n_par, mu]
        if parameters.dim() == 2:  # [n_par, mu]
            B = x.shape[1]
            parameters = parameters.unsqueeze(0).expand(B, -1, -1)
        elif parameters.dim() == 3:  # [B, n_par, mu]
            B = parameters.shape[0]
        else:
            raise ValueError("parameters must be [B, n_par, mu] or [n_par, mu].")

        # Clamp params to [0,1] before scaling
        parameters = torch.clamp(parameters, 0.0, 1.0)

        # Initialization
        if instate:
            assert init_states is not None and len(init_states) == 5, \
                "init_states must be (SNOWPACK, MELTWATER, SM, SUZ, SLZ)"
            SNOWPACK, MELTWATER, SM, SUZ, SLZ = [t.to(device) for t in init_states]
            # Clamp to keep stable
            SM  = torch.clamp(SM,  min=PRECS)
            SUZ = torch.clamp(SUZ, min=PRECS)
            SLZ = torch.clamp(SLZ, min=PRECS)
            start_t = 0
        elif bufftime > 0:
            with torch.no_grad():
                xinit = x[0:bufftime, :, :]
                initmodel = HBVMulET_1_1p()
                Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = initmodel(
                    xinit, parameters, mu, muwts, rtwts,
                    bufftime=0, outstate=True, routOpt=False, comprout=False
                )
            start_t = bufftime
        else:
            B = x.shape[1]
            SNOWPACK = (torch.zeros([B, mu], dtype=torch.float32, device=device) + 0.001)
            MELTWATER = (torch.zeros([B, mu], dtype=torch.float32, device=device) + 0.001)
            SM        = (torch.zeros([B, mu], dtype=torch.float32, device=device) + 0.001)
            SUZ       = (torch.zeros([B, mu], dtype=torch.float32, device=device) + 0.001)
            SLZ       = (torch.zeros([B, mu], dtype=torch.float32, device=device) + 0.001)
            start_t = 0

        # Forcings
        P     = x[start_t:, :, 0]
        T     = x[start_t:, :, 1]
        ETpot = x[start_t:, :, 2]
        Pm   = P.unsqueeze(2).repeat(1, 1, mu)
        Tm   = T.unsqueeze(2).repeat(1, 1, mu)
        ETpm = ETpot.unsqueeze(2).repeat(1, 1, mu)

        # Parameter ranges
        parascaLst = [
            [1, 6], [50, 1000], [0.05, 0.9], [0.01, 0.5],
            [0.001, 0.2], [0.2, 1], [0, 10], [0, 100],
            [-2.5, 2.5], [0.5, 10], [0, 0.1], [0, 0.2],
            [0.3, 5], [0, 1],
        ]
        routscaLst = [[0, 2.9], [0, 6.5]]

        # Scale parameters
        parBETA   = parascaLst[0][0]  + parameters[:, 0, :]  * (parascaLst[0][1]  - parascaLst[0][0])
        parFC     = parascaLst[1][0]  + parameters[:, 1, :]  * (parascaLst[1][1]  - parascaLst[1][0])
        parK0     = parascaLst[2][0]  + parameters[:, 2, :]  * (parascaLst[2][1]  - parascaLst[2][0])
        parK1     = parascaLst[3][0]  + parameters[:, 3, :]  * (parascaLst[3][1]  - parascaLst[3][0])
        parK2     = parascaLst[4][0]  + parameters[:, 4, :]  * (parascaLst[4][1]  - parascaLst[4][0])
        parLP     = parascaLst[5][0]  + parameters[:, 5, :]  * (parascaLst[5][1]  - parascaLst[5][0])
        parPERC   = parascaLst[6][0]  + parameters[:, 6, :]  * (parascaLst[6][1]  - parascaLst[6][0])
        parUZL    = parascaLst[7][0]  + parameters[:, 7, :]  * (parascaLst[7][1]  - parascaLst[7][0])
        parTT     = parascaLst[8][0]  + parameters[:, 8, :]  * (parascaLst[8][1]  - parascaLst[8][0])
        parCFMAX  = parascaLst[9][0]  + parameters[:, 9, :]  * (parascaLst[9][1]  - parascaLst[9][0])
        parCFR    = parascaLst[10][0] + parameters[:,10, :]  * (parascaLst[10][1] - parascaLst[10][0])
        parCWH    = parascaLst[11][0] + parameters[:,11, :]  * (parascaLst[11][1] - parascaLst[11][0])
        parBETAET = parascaLst[12][0] + parameters[:,12, :]  * (parascaLst[12][1] - parascaLst[12][0])
        parC      = parascaLst[13][0] + parameters[:,13, :]  * (parascaLst[13][1] - parascaLst[13][0])

        Nstep, Ngrid = P.size()
        Qsimmu = (torch.zeros(Pm.size(), dtype=torch.float32, device=device) + 0.001)

        for t in range(Nstep):
            PRECIP = Pm[t, :, :]
            RAIN = torch.mul(PRECIP, (Tm[t, :, :] >= parTT).type(torch.float32))
            SNOW = torch.mul(PRECIP, (Tm[t, :, :] < parTT).type(torch.float32))

            # Snow
            SNOWPACK = SNOWPACK + SNOW
            melt = torch.clamp(parCFMAX * (Tm[t, :, :] - parTT), min=0.0)
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK  = SNOWPACK - melt
            refreezing = torch.clamp(parCFR * parCFMAX * (parTT - Tm[t, :, :]), min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK   = SNOWPACK + refreezing
            MELTWATER  = MELTWATER - refreezing
            tosoil = torch.clamp(MELTWATER - (parCWH * SNOWPACK), min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil & ET
            soil_wetness = torch.clamp((SM / parFC) ** parBETA, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness
            SM = SM + RAIN + tosoil - recharge
            excess = torch.clamp(SM - parFC, min=0.0)
            SM = SM - excess
            evapfactor = torch.clamp((SM / (parLP * parFC)) ** parBETAET, min=0.0, max=1.0)
            ETact = torch.min(SM, ETpm[t, :, :] * evapfactor)
            SM = torch.clamp(SM - ETact, min=PRECS)

            # Clamp state to avoid negatives
            SM  = torch.clamp(SM,  min=PRECS)
            SUZ = torch.clamp(SUZ, min=PRECS)
            SLZ = torch.clamp(SLZ, min=PRECS)

            # Groundwater
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, parPERC)
            SUZ  = SUZ - PERC
            Q0 = parK0 * torch.clamp(SUZ - parUZL, min=0.0); SUZ = SUZ - Q0
            Q1 = parK1 * SUZ;                                 SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = parK2 * SLZ;                                 SLZ = SLZ - Q2

            Qsimmu[t, :, :] = Q0 + Q1 + Q2

        # Average components
        Qsimave = Qsimmu.mean(-1) if muwts is None else (Qsimmu * muwts).sum(-1)

        # Optional routing
        if routOpt:
            if comprout:
                Qsim = Qsimmu.view(Nstep, Ngrid * mu)
            else:
                Qsim = Qsimave
            if rtwts is None:
                raise ValueError("routOpt=True requires rtwts with shape [B,2].")
            rtwts = torch.clamp(rtwts, 0.0, 1.0)
            tempa = routscaLst[0][0] + rtwts[:, 0] * (routscaLst[0][1] - routscaLst[0][0])
            tempb = routscaLst[1][0] + rtwts[:, 1] * (routscaLst[1][1] - routscaLst[1][0])
            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            UH = UH_gamma(routa, routb, lenF=15)
            rf = torch.unsqueeze(Qsim, -1).permute(1, 2, 0)
            UH = UH.permute(1, 2, 0)
            Qsrout = UH_conv(rf, UH).permute(2, 0, 1)
            Qs = Qsrout
        else:
            Qs = torch.unsqueeze(Qsimave, -1)  # [T,B,1]

        return (Qs, SNOWPACK, MELTWATER, SM, SUZ, SLZ) if outstate else Qs
