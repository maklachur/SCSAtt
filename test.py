from __future__ import absolute_import

from got10k.experiments import *

from siamfc import TrackerSiamFC
from config import config
import os
import glob
import numpy as np

if __name__ == '__main__':

    # setup the tracker to access the pre-trained model
    folder_path = 'model'
    results = 'results'
    reports = 'reports'
    model=np.sort(glob.glob(os.path.join(folder_path,"*.pth")))

    for i in model:
        model_name = os.path.splitext(os.path.basename(i))[0]
        
        results_path_bbox = os.path.join(results,model_name)
        reports_path_graph = os.path.join(reports,model_name)

        tracker_test = TrackerSiamFC(net_path=i)
        experiments = ExperimentOTB(config.OTB_dataset_directoty, version=2015,
                                    result_dir=results_path_bbox,
                                    report_dir=reports_path_graph)
    
    
        # run the experiments for tracking to report the performance
        experiments.run(tracker_test, visualize=False)
        experiments.report([tracker_test.name])
