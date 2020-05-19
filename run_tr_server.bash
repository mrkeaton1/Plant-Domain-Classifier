for pt in True False
do
	for lr in 0.01 0.05 0.1 0.25 0.5 0.75
	do
		for mom in 0.5 0.7 0.9 0.95
		do
			python3 train_resnet.py "/data/mrkeaton/Datasets/Annotated iNaturalist Dataset" resnet-18 $pt 512 512 25 $lr $mom
			python3 train_resnet.py "/data/mrkeaton/Datasets/Annotated iNaturalist Dataset" resnet-34 $pt 256 256 25 $lr $mom
		done
	done
done

