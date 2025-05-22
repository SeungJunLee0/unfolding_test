import ROOT

# 파일 열기
f_bayes = ROOT.TFile.Open("unfold_top_mass_bayesian.root")
f_svd   = ROOT.TFile.Open("unfold_top_mass_svd.root")
f_tunf  = ROOT.TFile.Open("unfold_top_mass_TUnfolding.root")

# 히스토그램 로딩
h_truth_bayes = f_bayes.Get("hTrue")
h_truth_svd   = f_svd.Get("hTrue")
h_truth_tunf  = f_tunf.Get("hTrue")  # 각 파일에 hTrue 있음

# 비교를 위해 같은 binning이 사용되었는지 확인 필요 (assumed 동일)
h_bayes = f_bayes.Get("hUnfolded_Bayes")
h_svd   = f_svd.Get("hUnfoldedSVD")
h_tunf  = f_tunf.Get("hUnfoldedTopMass")

# Chi2/NDF 계산 (Unweighted, normalized)
print("Unfolding Comparison (χ²/NDF to Truth)")
print("--------------------------------------")

chi2_bayes = h_bayes.Chi2Test(h_truth_bayes, "UW CHI2/NDF")
chi2_svd   = h_svd.Chi2Test(h_truth_svd, "UW CHI2/NDF")
chi2_tunf  = h_tunf.Chi2Test(h_truth_tunf, "UW CHI2/NDF")

print(f"Bayesian  : χ²/NDF = {chi2_bayes:.3f}")
print(f"SVD       : χ²/NDF = {chi2_svd:.3f}")
print(f"TUnfold   : χ²/NDF = {chi2_tunf:.3f}")

c = ROOT.TCanvas("c", "Unfolded Results", 800, 600)
h_truth_bayes.SetLineColor(ROOT.kBlack)
h_bayes.SetLineColor(ROOT.kBlue)
h_svd.SetLineColor(ROOT.kRed)
h_tunf.SetLineColor(ROOT.kGreen+2)

h_truth_bayes.SetLineWidth(2)
h_bayes.SetLineWidth(2)
h_svd.SetLineWidth(2)
h_tunf.SetLineWidth(2)

h_truth_bayes.SetTitle("Truth vs Unfolded")
h_truth_bayes.GetXaxis().SetTitle("Top Mass [GeV]")
h_truth_bayes.GetYaxis().SetTitle("Events")
h_truth_bayes.Draw("hist")
h_bayes.Draw("hist same")
h_svd.Draw("hist same")
h_tunf.Draw("hist same")

legend = ROOT.TLegend(0.6, 0.7, 0.88, 0.88)
legend.AddEntry(h_truth_bayes, "Truth", "l")
legend.AddEntry(h_bayes, "Bayesian", "l")
legend.AddEntry(h_svd, "SVD", "l")
legend.AddEntry(h_tunf, "TUnfold", "l")
legend.Draw()

c.SaveAs("compare_unfolding_methods.png")

