from dataSets import loadDataSet , Config,classedCount
import torch
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.training import GDumb
from avalanche.models import IcarlNet, make_icarl_net, initialize_icarl_net
from avalanche.models import SimpleMLP, as_multitask
from torch.optim import SGD , Adam
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    ExperienceAccuracy,
    StreamAccuracy,
    EpochAccuracy,
)
from avalanche.logging.interactive_logging import InteractiveLogger
from torch.optim.lr_scheduler import MultiStepLR
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
import time
import pandas as pd


### Cpu / Gpu
### ### ### ### ### ### ### ### ### ### ### ### ### ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



### Parameters
### ### ### ### ### ### ### ### ### ### ### ### ### ###
dataSetNams = ["audioCub","ESC10","ESC50"]

def runGDUMB(config,BaseSaveDir,runningIdx = 0):

    train_data, nb_exp = loadDataSet(dataSetNams[config['idxOfUsedDataSet']],
                                     "train", config['applyBallanced'],
                                     config['FRAME_SIZE'],
                                     config['HOP_LENGTH'],
                                     config['N_MELS'])
    test_data, nb_exp = loadDataSet(dataSetNams[config['idxOfUsedDataSet']],
                                    "test",
                                    config['applyBallanced'],
                                    config['FRAME_SIZE'],
                                    config['HOP_LENGTH'],
                                    config['N_MELS'])

    train_data_avalanche = AvalancheDataset(train_data)
    test_data_avalanche = AvalancheDataset(test_data)
    data_tensor, label = test_data[0]

    ###  The benchmark
    ### ### ### ### ### ### ### ### ### ### ### ### ### ###
    scenario = nc_benchmark(
        train_dataset=train_data,
        test_dataset=test_data,
        n_experiences=config["nb_exp"],
        task_labels=False,
        seed=config['seed'],
        shuffle=False,
    )


    def transformData(inp):
        return inp



    ### Evaluation , plugins, and loggers
    ### ### ### ### ### ### ### ### ### ### ### ### ###

    evaluator = EvaluationPlugin(
          EpochAccuracy(),
          ExperienceAccuracy(),
          StreamAccuracy(),
          loggers=[InteractiveLogger()],
      )

    ### The Model
    ### ### ### ### ### ### ### ### ### ### ### ### ###

    if(config["modelType"] == "icarlNet"):
        model = make_icarl_net(num_classes=config['nb_exp'],n=5, c=1)
        model.apply(initialize_icarl_net)
    else:

        model = SimpleMLP(num_classes=config['nb_exp'],
                          input_size=data_tensor.size()[1] * data_tensor.size()[2],
                          hidden_size=config['model_hs'],
                          drop_rate=0)

    if (config['opt'] == "ADAM"):
        optim = Adam(
            model.parameters(),
            lr=config['lr_base'],
            weight_decay=config['wght_decay'],
        )



    else:
        optim = SGD(
            model.parameters(),
            lr=config['lr_base'],
            weight_decay=config['wght_decay'],
            momentum=config['momentum'],
        )

    sched = LRSchedulerPlugin(
        MultiStepLR(optim, [config['lr_milestones_1'], config['lr_milestones_2']], gamma=1.0 / config['lr_factor'])
    )


    ### the strategy
    ### ### ### ### ### ### ### ### ### ### ### ### ### ###



    strategy =  GDumb(
            model,
            optim,
            torch.nn.CrossEntropyLoss(),
            mem_size=config["replay_buffer_size"],
            train_mb_size=config["train_batch_size"] ,
            train_epochs=config["epochs"],
            eval_mb_size=config["eval_batch_size"],
            evaluator=evaluator,
            plugins=[sched],
            device=device)


    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
    ###  Training Code
    print("Starting experiment...")
    evaluationResults = []
    processingTime = []
    startingTime = time.time()
    print("startingTime : ", startingTime)
    for i, exp in enumerate(scenario.train_stream):
        eval_exps = [e for e in scenario.test_stream][: i + 1]

        print("Start training on experience ", exp.current_experience)
        expTime = time.time()
        print("starting a new exp time : ", expTime)
        val_exps = [e for e in scenario.test_stream][: i + 1]
        strategy.train(exp, num_workers=4)
        print("End training on experience", exp.current_experience)

        evaluationResults.append(strategy.eval(eval_exps, num_workers=4))
        print("Computing accuracy on the test set")
        expEndTime = time.time()
        expDuration = expEndTime - expTime
        processingTime.append(expDuration)

    endTime = time.time()
    print("endTime : ", endTime)
    print("total Processing time :", endTime - startingTime)
    procTime = endTime - startingTime


    num_classes = config['nb_exp']
    evaluationDfColumns = ['Exp', 'testAcc', 'trainAcc']
    lst = range(0, num_classes)
    evaluationDfColumns = evaluationDfColumns + ["testAccExp{:d}".format(x) for x in lst]

    evaluationDf = pd.DataFrame(columns=evaluationDfColumns)
    trainingDf = pd.DataFrame(columns=evaluationDfColumns)

    for i in range(0, num_classes):

        evaluationLine = {}
        evaluationLine = {'Exp': i,
                          'testAcc': evaluationResults[i]['Top1_Acc_Stream/eval_phase/test_stream/Task000'],
                          'trainAcc': evaluationResults[i]['Top1_Acc_Epoch/train_phase/train_stream/Task000']}

        for j in range(0, num_classes):
            if (j <= i):
                evaluationLine["testAccExp{:d}".format(j)] = evaluationResults[i][
                    "Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{:03d}".format(j)]

        new_row = pd.Series(data=evaluationLine)

        evaluationDf = evaluationDf.append(new_row, ignore_index=True)

    evaluationDf.to_csv(BaseSaveDir + "Gdumb_{:d}.csv".format(runningIdx), index=False)

    print("Ending the  seed ", config["seed"])
    print("last set training acc:", evaluationDf['trainAcc'].iloc[-1])
    print("training acc:", evaluationDf['trainAcc'].mean())
    print("testing acc:", evaluationDf['testAcc'].iloc[-1])

    config['trainAccMean'] = evaluationDf['trainAcc'].mean()
    config['trainAccLast'] = evaluationDf['trainAcc'].iloc[-1]
    config['testAcc'] = evaluationDf['testAcc'].iloc[-1]
    config['procTime'] = procTime
    configDF = pd.DataFrame(config, index=[0])
    configDF.to_csv(BaseSaveDir + "GDUMB_Config_{:d}.csv".format(runningIdx), index=False)

    return evaluationDf['trainAcc'].mean(), evaluationDf['trainAcc'].iloc[-1], evaluationDf['testAcc'].iloc[-1], procTime