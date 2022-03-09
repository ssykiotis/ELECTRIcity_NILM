# import argparse
from config        import *
from UKDALE_Parser import *
from REDD_Parser   import *

if __name__ == "__main__":

    args = get_args()
    setup_seed(args.seed)


    if args.dataset_code == 'redd_lf':
        args.house_indicies = [2, 3, 4, 5, 6]
        ds_parser = Redd_Parser(args)
    elif args.dataset_code == 'uk_dale':
        args.house_indicies = [1, 3, 4, 5]
        ds_parser = UK_Dale_Parser(args)

    # model = ELECTRICITY(args)
    # trainer = Trainer(args,model,stats) #is stats necessary?

    # #Training Loop
    # start_time = time()
    # if args.num_epochs > 0:
    # try:
    #     model.load_state_dict(torch.load(os.path.join(
    #         args.export_root, 'best_acc_model.pth'), map_location='cpu'))
    #     print('Successfully loaded previous model, continue training...')
    # except FileNotFoundError:
    #     print('Failed to load old model, continue training new model...')
    # trainer.train()

    # end_time = time()

    # training_time = end_time-start_time
    # print("Total Training Time: " + str(training_time/60) + "minutes")

    # #Testing Loop
    # args.validation_size = 1.
    # if args.dataset_code == 'redd_lf':
    #     args.house_indicies = [1]
    #     dataset = REDD_LF_Dataset(args, stats)
    # elif args.dataset_code == 'uk_dale':
    #     args.house_indicies = [2]
    #     dataset = UK_DALE_Dataset(args, stats)

    # dataloader = NILMDataloader(args, dataset)
    # _, test_loader = dataloader.get_dataloaders()
    # rel_err, abs_err, acc, prec, recall, f1 = trainer.test(test_loader)
    # print('Mean Accuracy:', acc)
    # print('Mean F1-Score:', f1)
    # print('Mean Relative Error:', rel_err)
    # print('Mean Absolute Error:', abs_err)

    # results = dict()

    # results['args']          = args
    # results['training_time']  = training_time/60
    # results['best_epoch']    = trainer.best_model_epoch 
    # results['training_loss'] = trainer.training_loss
    # results['val_rel_err']   = trainer.val_rel_err
    # results['val_abs_err']   = trainer.val_abs_err
    # results['val_acc']       = trainer.val_acc
    # results['val_precision'] = trainer.val_precision
    # results['val_recall']    = trainer.val_recall
    # results['val_f1']        = trainer.val_f1

    # results['label_curve']   = trainer.label_curve
    # results['e_pred_curve']  = trainer.e_pred_curve
    # results['status_curve']  = trainer.status_curce
    # results['s_pred_curve']  = trainer.s_pred_curve

    # import pickle as pkl
    # fname = Path(args.export_root).joinpath('results.pkl')
    # pkl.dump(results,open( fname, "wb" )) 

