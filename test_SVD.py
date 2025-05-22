#!/usr/bin/env python3
import ROOT, random, math
import numpy as np
from scipy.linalg import svd
import json

def crystal_ball_pdf_lefttail(x, par):
    alpha = par[0]
    n     = par[1]
    mean  = par[2]
    sigma = par[3]
    t = (x[0] - mean) / sigma
    if t > - alpha:
        return ROOT.TMath.Exp(-0.5 * t*t)
    else:
        A = (n/alpha)**n * ROOT.TMath.Exp(-0.5 * alpha*alpha)
        B = (n/alpha) - alpha
        return A * (B - t)**(-n)

def hist_to_numpy(hist):
    """ROOT histogram을 numpy array로 변환 (1D histo)"""
    nbins = hist.GetNbinsX()
    arr = np.array([hist.GetBinContent(i+1) for i in range(nbins)])
    return arr

def hist2d_to_numpy(hist2d):
    """ROOT 2D histogram을 numpy matrix로 변환"""
    nbins_x = hist2d.GetNbinsX()
    nbins_y = hist2d.GetNbinsY()
    mat = np.zeros((nbins_y, nbins_x))
    for i in range(nbins_x):
        for j in range(nbins_y):
            mat[j, i] = hist2d.GetBinContent(i+1, j+1)
    return mat

def numpy_to_hist(arr, name, title, x_min, x_max):
    """numpy array를 ROOT 1D histogram으로 변환"""
    nbins = len(arr)
    hist = ROOT.TH1F(name, title, nbins, x_min, x_max)
    for i in range(nbins):
        hist.SetBinContent(i+1, arr[i])
    return hist


def load_txt_config(file_path):
    config = {}
    with open(file_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=')
                key = key.strip()
                val = value.strip()
                try:
                    config[key] = int(val)
                except ValueError:
                    try:
                        config[key] = float(val)
                    except ValueError:
                        config[key] = val
    return config


def main():
    # ---------- 파라미터 설정 ----------
    config = load_txt_config("config.txt")
    N_EVENTS = config["N_EVENTS"]       # toy MC 이벤트 수
    TOPMEAN  = config["TOPMEAN"]        # true top mass 평균
    TOPSIGMA = config["TOPSIGMA"]       # true top mass 폭 (가우시안)

    # 오른쪽 꼬리 Crystal Ball 파라미터
    RES_SIGMA = config["RES_SIGMA"]     # 측정 분포의 가우시안 폭
    ALPHA     = config["ALPHA"]         # 테일 시작점
    N_tail    = config["N_tail"]        # 꼬리의 폴리노미얼 차수

    x_min, x_max = 130.0, 210.0
    nbins = 40

    # ---------- 히스토그램 준비 ----------
    hTrue = ROOT.TH1F("hTrue", "True Top Mass", nbins, x_min, x_max)
    hMeas = ROOT.TH1F("hMeas", "Measured Top Mass", nbins, x_min, x_max)
    hResponse2D = ROOT.TH2F("hResponse2D", "Response Matrix", nbins, x_min, x_max, nbins, x_min, x_max)

    # ---------- TF1 생성: 왼쪽 꼬리 CB ----------
    myCB = ROOT.TF1("myCB_lefttail", crystal_ball_pdf_lefttail, -50, 50, 4)
    myCB.SetParameters(ALPHA, N_tail, 0.0, RES_SIGMA)
    myCB.SetNpx(3000)

    # ---------- toy 이벤트 생성 ----------
    for _ in range(N_EVENTS):
        m_true = random.gauss(TOPMEAN, TOPSIGMA)
        delta = myCB.GetRandom()
        m_reco = m_true + delta

        if not (x_min < m_true < x_max):
            continue
        if not (x_min < m_reco < x_max):
            continue

        hTrue.Fill(m_true)
        hMeas.Fill(m_reco)
        hResponse2D.Fill(m_true, m_reco)

    # ---------- numpy로 변환 ----------
    y_meas = hist_to_numpy(hMeas)  # measured distribution
    R = hist2d_to_numpy(hResponse2D)  # response matrix
    col_sums = R.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0  # 0인 column은 1로 바꿔서 나누기
    R_norm = R / col_sums
#    R_norm = R / R.sum(axis=0, keepdims=True)  # response matrix 정규화 (col 방향으로)

    # ---------- SVD 분해 ----------
    U, s, Vt = svd(R_norm, full_matrices=False)
    print("SVD 완료: U shape", U.shape, "Σ shape", s.shape, "Vt shape", Vt.shape)

    # ---------- Regularization ----------
    # 작은 특이값에 대한 damping 적용
    tau = 0.01  # Regularization strength (너가 조정할 수 있음)
    s_inv = s / (s**2 + tau**2)

    # unfolded solution 계산
    x_unfold = Vt.T @ np.diag(s_inv) @ U.T @ y_meas

    # ---------- 결과 ROOT 히스토그램 변환 ----------
    hUnfolded = numpy_to_hist(x_unfold, "hUnfoldedSVD", "Unfolded Top Mass (SVD)", x_min, x_max)

    # ---------- 결과 출력 ----------
    print("Generated events:", N_EVENTS)
    print("True integral:", hTrue.Integral())
    print("Measured integral:", hMeas.Integral())
    print("Unfolded (SVD) integral:", hUnfolded.Integral())

    # ---------- 캔버스 ----------
    c1 = ROOT.TCanvas("c1", "True, Meas, Unfolded (SVD)", 1200, 600)
    hTrue.SetLineColor(ROOT.kBlue)
    hTrue.SetTitle("True,Reco,Unfolded(SVD)")
    hTrue.Draw("HIST")
    hMeas.SetLineColor(ROOT.kRed)
    hMeas.Draw("HIST SAME")
    hUnfolded.SetLineColor(ROOT.kGreen+2)
    hUnfolded.Draw("HIST SAME")

    # ---------- 저장 ----------
    outFile = ROOT.TFile("unfold_top_mass_svd.root", "RECREATE")
    hTrue.Write()
    hMeas.Write()
    hResponse2D.Write()
    hUnfolded.Write()
    c1.Write()
    outFile.Close()
    print("모든 객체가 unfold_top_mass_svd.root 에 저장되었습니다.")

if __name__ == "__main__":
    main()

