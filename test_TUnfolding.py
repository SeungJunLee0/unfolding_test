#!/usr/bin/env python3
import ROOT, random, math, numpy as np
import json

def crystal_ball_pdf_lefttail(x, par):
    """
    Crystal Ball 함수 (left tail 버전)
    
    par = [alpha, n, mean, sigma]
      - alpha : 테일 시작 지점 (>0)
      - n     : 테일에서의 폴리노미얼 차수 (>0)
      - mean  : 중심값
      - sigma : 가우시안 부분의 폭
    x[0]는 현재 x 값
    """
    alpha = par[0]
    n     = par[1]
    mean  = par[2]
    sigma = par[3]
    t = (x[0] - mean) / sigma
    if t > -alpha:
        return ROOT.TMath.Exp(-0.5 * t*t)
    else:
        A = (n/alpha)**n * ROOT.TMath.Exp(-0.5 * alpha*alpha)
        B = (n/alpha) - alpha
        return A * (B - t)**(-n)


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

    # 히스토그램 범위 및 bin 수
    x_min, x_max = 130.0, 210.0
    nbins = 40

    # ---------- 히스토그램 준비 ----------
    hTrue = ROOT.TH1F("hTrue", "True Top Mass", nbins, x_min, x_max)
    hMeas = ROOT.TH1F("hMeas", "Measured (Right-tail CB) Top Mass", nbins, x_min, x_max)
    hResponse2D = ROOT.TH2F("hResponse2D", "Migration Matrix; True M_{top}; Reco M_{top}",
                            nbins, x_min, x_max, nbins, x_min, x_max)

    # ---------- TF1 생성: 오른쪽 꼬리 CB ----------
    myCB = ROOT.TF1("myCB_lefttail", crystal_ball_pdf_lefttail, -50, 50, 4)
    myCB.SetParameters(ALPHA, N_tail, 0.0, RES_SIGMA)  # mean=0, sigma=RES_SIGMA
    myCB.SetNpx(3000)  # GetRandom() 정확도 향상

    # ---------- toy 이벤트 생성 (응답 행렬) ----------
    for _ in range(N_EVENTS):
        # (1) true mass: 가우시안 분포
        m_true = random.gauss(TOPMEAN, TOPSIGMA)
        # (2) 측정값: true mass에 오른쪽 꼬리 CB에서 생성한 델타를 더함
        delta = myCB.GetRandom()
        m_reco = m_true + delta

        # 범위 검사
        if not (x_min < m_true < x_max):
            continue
        if not (x_min < m_reco < x_max):
            continue

        hTrue.Fill(m_true)
        hMeas.Fill(m_reco)
        hResponse2D.Fill(m_true, m_reco)

    # ---------- TUnfold로 언폴딩 ----------
    # 여기서는 kHistMapOutputHoriz 옵션을 사용합니다.
    unfold = ROOT.TUnfold(hResponse2D, ROOT.TUnfold.kHistMapOutputHoriz)
    ret_code = unfold.SetInput(hMeas)
    if ret_code >= 10000:
        print(f"[WARN] TUnfold SetInput returned code {ret_code}")

    # ---------- τ 스캔을 통한 L-curve 스캔 ----------
    tau_min = 1e-3
    tau_max = 1e3
    n_steps = 50
    taus = np.logspace(math.log10(tau_min), math.log10(tau_max), n_steps)
    chi2_vals = []
    reg_vals = []

    for tau in taus:
        # 주어진 τ로 unfolding 수행
        unfold.DoUnfold(tau)
        # 기존 GetChi2() 대신 GetGlobalChi2() 사용 (버전에 따라 이 함수가 χ²를 계산함)
        chi2 = unfold.GetChi2A()
        reg  = unfold.GetChi2L()
        chi2_vals.append(chi2)
        reg_vals.append(reg)
        print(f"tau = {tau:.3e}: chi2 = {chi2:.2f}, reg = {reg:.2f}")

    # 최적의 τ 선택 (여기서는 chi2 + reg의 합이 최소인 τ 선택; 실제 분석에서는 더 정교한 기준 사용 가능)
    total = np.array(chi2_vals) + np.array(reg_vals)
    idx_best = int(np.argmin(total))
    tau_best = taus[idx_best]
    print("선택된 최적 τ =", tau_best)

    # 최적의 τ로 다시 언폴딩
    unfold.DoUnfold(tau_best)
    hUnfolded = ROOT.TH1F("hUnfoldedTopMass", "Unfolded Distribution", nbins, x_min, x_max)
    unfold.GetOutput(hUnfolded, 0)

    # ---------- 결과 출력 ----------
    print("Generated events:", N_EVENTS)
    print("True integral:", hTrue.Integral())
    print("Measured integral:", hMeas.Integral())
    print("Unfolded integral:", hUnfolded.Integral())

    # 캔버스 1: 개별 분포 그리기
    c1 = ROOT.TCanvas("c1", "True, Meas, Unfolded", 1200, 600)
    c1.Divide(3,1)
    c1.cd(1)
    hTrue.SetLineColor(ROOT.kBlue)
    hTrue.SetTitle("True Distribution")
    hTrue.Draw("HIST")
    c1.cd(2)
    hMeas.SetLineColor(ROOT.kRed)
    hMeas.SetTitle("Measured Distribution")
    hMeas.Draw("HIST")
    c1.cd(3)
    hUnfolded.SetLineColor(ROOT.kGreen+2)
    hUnfolded.SetTitle("Unfolded Distribution")
    hUnfolded.Draw("HIST")

    # 캔버스 2: L-curve 스캔 그래프 (τ vs. chi2 및 reg term)
    c2 = ROOT.TCanvas("c2", "L-curve Scan", 800, 600)
    graph_chi2 = ROOT.TGraph(n_steps, np.array(taus, dtype=float), np.array(chi2_vals, dtype=float))
    graph_reg = ROOT.TGraph(n_steps, np.array(taus, dtype=float), np.array(reg_vals, dtype=float))
    graph_chi2.SetLineColor(ROOT.kRed)
    graph_chi2.SetLineWidth(2)
    graph_reg.SetLineColor(ROOT.kBlue)
    graph_reg.SetLineWidth(2)
    graph_chi2.SetTitle("L-curve Scan; #tau; #chi^{2} and Reg Term")
    graph_chi2.GetXaxis().SetMoreLogLabels()
    graph_chi2.GetXaxis().SetLimits(tau_min, tau_max)
    graph_chi2.Draw("AL")
    graph_reg.Draw("L SAME")
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(graph_chi2, "#chi^{2}", "l")
    legend.AddEntry(graph_reg, "Regularisation Term", "l")
    legend.Draw()
    c2.SetLogx()


    # 캔버스 3: 모든 분포 그리기
    c3 = ROOT.TCanvas("c3", "True, Meas, Unfolded", 1200, 600)
    hTrue.SetLineColor(ROOT.kBlue)
    hTrue.SetTitle("True,Reco,unfold Distribution")
    hTrue.Draw("HIST")
    hMeas.SetLineColor(ROOT.kRed)
    hMeas.Draw("HIST SAME")
    hUnfolded.SetLineColor(ROOT.kGreen+2)
    hUnfolded.Draw("HIST SAME")



    # ---------- 결과 ROOT 파일로 저장 ----------
    outFile = ROOT.TFile("unfold_top_mass_TUnfolding.root", "RECREATE")
    hTrue.Write()
    hMeas.Write()
    hResponse2D.Write()
    hUnfolded.Write()
    c1.Write()
    c2.Write()
    c3.Write()
    outFile.Close()
    print("모든 객체가 unfold_top_mass_TUnfolding.root 에 저장되었습니다.")

if __name__ == "__main__":
    main()

