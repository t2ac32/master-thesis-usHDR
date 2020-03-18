polyaxonmaster-thesis-usHDR

# Deep learning HDR DATASET

Bash usage command examples
```
python ushdrcnn_train.py --epoch=10 --batch-size=15 -g -n -t 
python ushdrcnn_train.py --epochs=10 --batch-size=15 --expo-num=15 -g -n -t

quick test:
python ushdrcnn_train.py --epochs=1 --batch-size=9 --expo-num=15 -g 


```
Check tensoroard Summary after training
```
tensorboard --logdir=runs
```

### Polyaxon

### Documentation
 - Hyper parameter search 
 	https://docs.polyaxon.com/concepts/experiment-groups-hyperparameters-optimization/
 - Grid search
  	https://docs.polyaxon.com/references/polyaxon-optimization-engine/grid-search/
#### Config local CLI

```
polyaxon config set --host=10.23.0.18 --port 31811
```
	If error: polyaxon: command not found 
	
	Try adding the python installation to the PATH
	```
	python3 -m site &> /dev/null && PATH="$PATH:`python3 -m site --user-base`/bin"
	```
#### Check polyaxonfile.yml
```
polyaxon check -f polyaxonfile.yaml
upload project:
polyaxon run -f polyaxonfile.yaml -u -l
```
where
- -u: upload/update project
- -f update polyaxonfile.yaml
- -l enable logs onto the standard output (the terminal)

### Check experiment:
``` 
http://cluster.ifl:31811
```

| Datasets                   | Inputs         | Input_imgs|
| ---------------------------|:-------------: | --------: |
| newDataset                 | 1901           | 28515     |
| LDR2_fakeComp_DataSet      | 26             |  390      |


## new_DataSet

**File Structure**
```Bash
├──Originals/
|  ├─Study/
|  ├─ date-generic/
|     |- 00.b8 -> 15.b8
├──images/
|  ├──Study/
|        | exVivo_xx.png
|        | exVivo_power.txt
|        ├─Results/
|           | - stack_hdr_image.hdr
|           | - hdrReinhard_global.png
|           | - hdrReinhard_local.png
```

## LDR_DataSet

**Folder Structure**
```Bash
├──Org_images/
|   ├── LDR_01469_0xxxx.tiff
├──c_images/
|   ├──LDR_0xxxx/
|       | exVivo_xx.png
|       | exVivo_power.txt
|       ├──Results/
|           | stack_hdr_image.hdr
|           | hdrReinhard_global.png
|           | hdrReinhard_local.png
```



## Fakehdr dataset v2

**biggest image:**    LDR_01469.tiff

  | maxH: 839 | maxH: 632  |
  | ----------|------------|

**Smallest image:**    LDR_00864.tiff

  | minH: 236 | minH: 279  |
  | ----------|------------|

**File Structure**

```Bash
├──Org_images/
|   ├── LDR_0xxxx.tiff
├──c_images/
|   ├── LDR_0xxxx/
|        | exVivo_xx.png
|        | exVivo_power.txt
|        ├──Results/
|           | stack_hdr_image.hdr
|           | hdrReinhard_global.png
|           | hdrReinhard_local.png
```

		| Training SETUP:       | Epoch finished ! 						|
		|Epochs: 1		| Train Loss:0.000266						|
		|Batch size: 15		| Val Loss:0.000046, running_val_loss:0.000046  		|
		|Learning rate: 0.1	| Validation loss: 0.000046188					|
		|Training size: 3042    | Training complete in 68m 48s					|
		| Validation size: 760  |								|
		|Checkpoints: False	|								|
		|CUDA: True		|								|	
		| ----------------------|---------------------------------------------------------------|



		| Training SETUP:       | Epoch finished ! 						|
		|Epochs: 8		| Train Loss:0.000274						|
		|Batch size: 15		| Val Loss:0.000049, running_val_loss:0.000049  		|
		|Learning rate: 0.1     | Validation loss: 0.000049238					|
		|Training size: 3042    | Training complete in 399m 26s					|
		| Validation size: 760  |								|
		|Checkpoints: False	|								|
		|CUDA: True		|								|	
		| ----------------------|---------------------------------------------------------------|



### Interesting links:

https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/
