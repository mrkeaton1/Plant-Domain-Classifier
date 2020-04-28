for pt in True False
do
	for lr in 0.01 0.05 0.1 0.25 0.5 0.75
	do
		for mom in 0.5 0.7 0.9 0.95
		do
			python3 train_on_mnist.py '/home/mrkeaton/Documents/PyTorch Learning Code/MNIST Digit Recognition' $pt 128 128 10 $lr $mom
		done
	done
done

