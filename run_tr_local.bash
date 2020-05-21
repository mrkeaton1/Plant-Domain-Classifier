for pt in True False
do
	for lr in 0.01 0.05 0.1 0.25 0.5 0.75
	do
		for mom in 0.5 0.7 0.9 0.95
		do
			python3 train_resnet.py "/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset - edited (new)" resnet-18 $pt 128 128 10 $lr $mom
			python3 train_resnet.py "/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset - edited (new)" resnet-34 $pt 64 64 10 $lr $mom
			python3 train_resnet.py "/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset - edited (new)" efficientnet-b0 $pt 32 32 10 $lr $mom
			python3 train_resnet.py "/home/mrkeaton/Documents/Datasets/Annotated iNaturalist Dataset - edited (new)" efficientnet-b7 $pt 4 4 10 $lr $mom
		done
	done
done

