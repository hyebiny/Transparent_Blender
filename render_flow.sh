# Use graycode pattern to obtain the ground truth refractive flow field
# mkdir -p ${calibDir}${obj}
# sed -e  's#\${ImageName}#"./data/graycode_512_512/graycode_"#' ${template} > ${setting}
# povray -I$setting $common +HI${dataDir}${obj}.inc ${cam_obj_bg_setting} +FN ${D}Calib=1 +KFI1 +KFF20 +KI1 +KF20 -O${calibDir}${obj}/graycode_
# python findCorrespondence.py --in_root ${calibDir} --in_dir ${obj} --out_dir ${outDir} 
python findCorrespondence.py --in_root './output_blender' --mute
