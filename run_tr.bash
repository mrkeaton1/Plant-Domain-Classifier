for pt in True False
do
	for lr in 0.01 0.05 0.1 0.25 0.5 0.75
	do
		for mom in 0.5 0.7 0.9 0.95
		do
			python3 train_resnet.py $pt 3 $lr $mom
		done
	done
done

