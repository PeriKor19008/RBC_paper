from __future__ import annotations
from src.utils.paths import rel_to_root
from src.model.experiments.tests_helper import *
LABEL_KEYS = ["diameter", "thickness", "ratio", "ref_index"]





def cor_run_single_label_ensemble():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Ref index new data
    ckpt_paths = [
        rel_to_root(
            "outputs/models/multyLabelCNN/refInex_newData/1,2-16,9(noise)35/ref_multyLabelCNN_ref_e35_lr0.001_bs64_val0.000255.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/refInex_newData/1,4-9,6(noise)35/ref_multyLabelCNN_ref_e35_lr0.001_bs64_val0.000208.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/refInex_newData/1,4-17,0(noise)45/ref_multyLabelCNN_ref_e45_lr0.001_bs64_val0.000202.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/refInex_newData/1,5-17,2(noise)41/ref_multyLabelCNN_ref_e41_lr0.001_bs64_val0.000198.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/refInex_newData/1,7-16,2(noise)36/ref_multyLabelCNN_ref_e36_lr0.001_bs64_val0.000254.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/refInex_newData/1,7-16,2(noise)41/ref_multyLabelCNN_ref_e41_lr0.001_bs64_val0.000175.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/refInex_newData/1,7-16,7(noise)35/ref_multyLabelCNN_ref_e35_lr0.001_bs64_val0.000248.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/refInex_newData/2,2-15,4(noise)43/ref_multyLabelCNN_ref_e43_lr0.001_bs64_val0.000194.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/refInex_newData/2,2-16,3(no_noise)21/ref_multyLabelCNN_ref_e21_lr0.001_bs64_val0.000446.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/refInex_newData/2,1-17,1(noise)48/ref_multyLabelCNN_ref_e48_lr0.001_bs64_val0.000215.pt"),
        rel_to_root(
            "outputs/models/multyLabelCNN/refInex_newData/1,7-18,0(noise)30/ref_multyLabelCNN_ref_e30_lr0.001_bs64_val0.000322.pt"),

    ]
    ### Ref Index ####
    # ckpt_paths = [
    #     rel_to_root(
    #         "outputs/models/multyLabelCNN/RefIndex/0,9-7,8/ref_multyLabelCNN_ref_e25_lr0.001_bs32_val0.000283.pt"),
    #     rel_to_root(
    #         "outputs/models/multyLabelCNN/RefIndex/1,2-6,4/ref_multyLabelCNN_ref_e43_lr0.001_bs32_val0.001069.pt"),
    #     rel_to_root(
    #         "outputs/models/multyLabelCNN/RefIndex/1,4-13,0/ref_multyLabelCNN_ref_e23_lr0.001_bs32_val0.000312.pt"),
    #     rel_to_root(
    #         "outputs/models/multyLabelCNN/RefIndex/1,6-8,6/ref_multyLabelCNN_ref_e37_lr0.001_bs32_val0.000381.pt"),
    #     rel_to_root(
    #         "outputs/models/multyLabelCNN/RefIndex/1,9-9,9/ref_multyLabelCNN_ref_e24_lr0.001_bs32_val0.000346.pt"),
    #     rel_to_root(
    #         "outputs/models/multyLabelCNN/RefIndex/1,1-22,3/ref_multyLabelCNN_ref_e44_lr0.001_bs32_val0.000560.pt"),
    #     rel_to_root(
    #         "outputs/models/multyLabelCNN/RefIndex/1,1-23/ref_multyLabelCNN_ref_e28_lr0.001_bs32_val0.000580.pt"),
    #     rel_to_root(
    #         "outputs/models/multyLabelCNN/RefIndex/1,7-10,5/ref_multyLabelCNN_ref_e25_lr0.001_bs32_val0.000257.pt"),
    #     rel_to_root(
    #         "outputs/models/multyLabelCNN/RefIndex/2,5-10,2/ref_multyLabelCNN_ref_e34_lr0.001_bs32_val0.000300.pt"),
    #     rel_to_root(
    #         "outputs/models/multyLabelCNN/RefIndex/2,1-10,8/ref_multyLabelCNN_ref_e39_lr0.001_bs32_val0.001834.pt"),
    #     rel_to_root(
    #         "outputs/models/multyLabelCNN/RefIndex/1,4-18,3/ref_multyLabelCNN_ref_e30_lr0.001_bs32_val0.000433.pt"),
    #
    # ]

    # ckpt_paths = [
    #     rel_to_root(
    #         "outputs/models/multyLabelCNN/RefIndex/0,9-7,8/ref_multyLabelCNN_ref_e25_lr0.001_bs32_val0.000283.pt"),
    #
    #     ]

    # 2. Load all models into a list
    models = []
    for path in ckpt_paths:
        print(f"Loading model: {path}")
        model = torch.load(path, map_location="cpu", weights_only=False).to(device).eval()
        models.append(model)

    data_dir_good = rel_to_root("Data/test_data/rs")
    out_pct = rel_to_root("outputs/test_graphs/single_ensemble_avg_pct_error.png")

    # 3. Choose which label this ensemble was trained for:
    # 0 = diameter, 1 = thickness, 2 = ratio, 3 = ref_index
    CURRENT_TARGET = 3

    # 4. Pass the LIST of models to the new testing function
    test_single_label_ensemble(
        models=models,
        dir_path=data_dir_good,
        save_path_pct=str(out_pct),
        thresh=99,
        block=False,
        jitter=False,
        noise=False,
        normalize=True,
        trimed=True,              # <--- Your trimmed mean logic is active!
        target_idx=CURRENT_TARGET,
        min_val=15# <--- Passed down to the test loop
    )

def single_label():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = rel_to_root(
        "outputs/models/multyLabelCNN/6,6-107,9(no_noise)74/ratio_multyLabelCNN_ratio_e74_lr0.001_bs64_val0.041430.pt")
    ae_path = rel_to_root(
        "outputs/models/FCAutoencoder/cor_noise_20251123-193317_FCAutoencoder_e25_lr0.001_bs32_wd0.0_seed42_dsmanual/autoencoder_final.pt")
    data_dir_good = rel_to_root("Data/test_data/rs")
    out_pct = rel_to_root("outputs/test_graphs/extra_runs_avg_pct_error.png")
    model = torch.load(ckpt_path, map_location="cpu", weights_only=False).to(device).eval()
    # ae_model =    torch.load(ae_path, map_location="cpu", weights_only=False).to(device).eval()
    ## "diameter", "thickness", "ratio", "ref_index"
    test_single_label(model,data_dir_good,str(out_pct),20,block=False,jitter=False,noise=False,normalize=True,target_idx=2)

if __name__ == "__main__":
    #print (a_infer_ref_index_from_path(Path("../../../Data/extra_runs_for_check/20_0737523754741a.f06")))

    #print(load_rbc_txt_image_and_labels("../../../Data/extra_runs_for_check/05_0362516257251a.f06"))

    # run_occlusion_demo(
    #     ckpt_path="outputs/models/FlexibleCNN/BEST_old6_20251104-070605_FlexibleCNN_e25_lr0.001_bs32_wd0.0_seed42_dsmanual/FlexibleCNN_e25_lr0.001_bs32_val0.004462.pt",
    #     sample_path="Data/extra_runs_good_img",per_label=False,avg=True,
    # )

    #cor_run_single_label_ensemble()
    single_label()


    #cor_run_ensemble()
    #cor_run()
    #run_grad_Cam()
    #frequency_test()







