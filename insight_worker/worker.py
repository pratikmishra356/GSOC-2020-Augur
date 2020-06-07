from multiprocessing import Process, Queue
from urllib.parse import urlparse
import requests, sys
import pandas as pd
import sqlalchemy as s
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import MetaData, and_
import statistics, logging, os, json, time
import numpy as np
import scipy.stats
import datetime
from sklearn.ensemble import IsolationForest
from workers.standard_methods import * #init_oauths, get_table_values, get_max_id, register_task_completion, register_task_failure, connect_to_broker, update_gh_rate_limit, record_model_process, paginate
from insight_worker.preprocess_metrics import preprocess_endpoints
import warnings
warnings.filterwarnings('ignore')

class InsightWorker:
    """ Worker that detects anomalies on a select few of our metrics
    task: most recent task the broker added to the worker's queue
    child: current process of the queue being ran
    queue: queue of tasks to be fulfilled
    config: holds info like api keys, descriptions, and database connection strings
    """
    def __init__(self, config, task=None):
        self.config = config
        logging.basicConfig(filename='worker_{}.log'.format(self.config['id'].split('.')[len(self.config['id'].split('.')) - 1]), filemode='w', level=logging.INFO)
        logging.info('Worker (PID: {}) initializing...\n'.format(str(os.getpid())))
        self._task = task
        self._child = None
        self._queue = Queue()
        self.db = None
        self.tool_source = 'Insight Worker'
        self.tool_version = '0.0.3' # See __init__.py
        self.data_source = 'Augur API'
        self.refresh = True
        self.send_insights = False
        self.finishing_task = False
        self.anomaly_days = 300#self.config['anomaly_days']
        self.training_days = self.config['training_days']
        self.contamination = self.config['contamination']
        self.confidence = self.config['confidence_interval'] / 100
        self.metrics = self.config['metrics']
        
        self.specs = {
            "id": self.config['id'],
            "location": self.config['location'],
            "qualifications":  [
                {
                    "given": [["git_url"]],
                    "models":["insights"]
                }
            ],
            "config": [self.config]
        }

        self.results_counter = 0

        self.DB_STR = 'postgresql://{}:{}@{}:{}/{}'.format(
            self.config['user'], self.config['password'], self.config['host'], self.config['port'], self.config['database']
        )

        dbschema = 'augur_data'
        self.db = s.create_engine(self.DB_STR, poolclass=s.pool.NullPool,
            connect_args={'options': '-csearch_path={}'.format(dbschema)})

        helper_schema = 'augur_operations'
        self.helper_db = s.create_engine(self.DB_STR, poolclass = s.pool.NullPool,
            connect_args={'options': '-csearch_path={}'.format(helper_schema)})
        
        # produce our own MetaData object
        metadata = MetaData()
        helper_metadata = MetaData()

        # we can reflect it ourselves from a database, using options
        # such as 'only' to limit what tables we look at...
        metadata.reflect(self.db, only=['chaoss_metric_status', 'repo_insights', 'repo_insights_records'])
        helper_metadata.reflect(self.helper_db, only=['worker_history', 'worker_job'])

        # we can then produce a set of mappings from this MetaData.
        Base = automap_base(metadata=metadata)
        HelperBase = automap_base(metadata=helper_metadata)

        # calling prepare() just sets up mapped classes and relationships.
        Base.prepare()
        HelperBase.prepare()

        # mapped classes are ready
        self.chaoss_metric_status_table = Base.classes['chaoss_metric_status'].__table__
        self.repo_insights_table = Base.classes['repo_insights'].__table__
        self.repo_insights_records_table = Base.classes['repo_insights_records'].__table__

        self.history_table = HelperBase.classes.worker_history.__table__
        self.job_table = HelperBase.classes.worker_job.__table__

        # Organize different api keys/oauths available
        init_oauths(self)

        # Send broker hello message
        connect_to_broker(self)

    def update_config(self, config):
        """ Method to update config and set a default
        """
        self.config = {
            'database_connection_string': 'psql://{}:5432/augur'.format(self.config['broker_host']),
            "display_name": "",
            "description": "",
            "required": 1,
            "type": "string"
        }
        self.config.update(config)

    @property
    def task(self):
        """ Property that is returned when the worker's current task is referenced
        """
        return self._task
    
    @task.setter
    def task(self, value):
        """ entry point for the broker to add a task to the queue
        Adds this task to the queue, and calls method to process queue
        """
        if value['job_type'] == "UPDATE" or value['job_type'] == "MAINTAIN":
            self._queue.put(value)
        
        self._task = value
        self.run()

    def cancel(self):
        """ Delete/cancel current task
        """
        self._task = None

    def run(self):
        """ Kicks off the processing of the queue if it is not already being processed
        Gets run whenever a new task is added
        """
        logging.info("Running...\n")
        self._child = Process(target=self.collect, args=())
        self._child.start()

    def collect(self):
        """ Function to process each entry in the worker's task queue
        Determines what action to take based off the message type
        """
        while True:
            if not self._queue.empty():
                message = self._queue.get() # Get the task off our MP queue
            else:
                break
            logging.info("Popped off message: {}\n".format(str(message)))

            if message['job_type'] == 'STOP':
                break

            # If task is not a valid job type
            if message['job_type'] != 'MAINTAIN' and message['job_type'] != 'UPDATE':
                raise ValueError('{} is not a recognized task type'.format(message['job_type']))
                pass

            # Query repo_id corresponding to repo url of given task 
            repoUrlSQL = s.sql.text("""
                SELECT min(repo_id) as repo_id FROM repo WHERE repo_git = '{}'
                """.format(message['given']['git_url']))
            repo_id = int(pd.read_sql(repoUrlSQL, self.db, params={}).iloc[0]['repo_id'])

            # Model method calls wrapped in try/except so that any unexpected error that occurs can be caught
            #   and worker can move onto the next task without stopping
            try:
                # Call method corresponding to model sent in task
                if message['models'][0] == 'insights':
                    self.insight_model(message, repo_id)
            except Exception as e:
                register_task_failure(self, message, repo_id, e)
                pass


    def insight_model(self,entry_info,repo_id):
        preprocess_endpoints(self,entry_info,repo_id)

        self.register_task_completion(entry_info, repo_id, "insights")

    


    def register_task_completion(self, entry_info, repo_id, model):
        # Task to send back to broker
        task_completed = {
            'worker_id': self.config['id'],
            'job_type': entry_info['job_type'],
            'repo_id': repo_id,
            'git_url': entry_info['git_url']
        }
        # Add to history table
        task_history = {
            "repo_id": repo_id,
            "worker": self.config['id'],
            "job_model": model,
            "oauth_id": self.config['zombie_id'],
            "timestamp": datetime.datetime.now(),
            "status": "Success",
            "total_results": self.results_counter
        }
        self.helper_db.execute(self.history_table.update().where(
            self.history_table.c.history_id==self.history_id).values(task_history))

        logging.info("Recorded job completion for: " + str(task_completed) + "\n")

        # Update job process table
        updated_job = {
            "since_id_str": repo_id,
            "last_count": self.results_counter,
            "last_run": datetime.datetime.now(),
            "analysis_state": 0
        }
        self.helper_db.execute(self.job_table.update().where(
            self.job_table.c.job_model==model).values(updated_job))
        logging.info("Update job process for model: " + model + "\n")

        # Notify broker of completion
        logging.info("Telling broker we completed task: " + str(task_completed) + "\n\n" + 
            "This task inserted: " + str(self.results_counter) + " tuples.\n\n")

        requests.post('http://{}:{}/api/unstable/completed_task'.format(
            self.config['broker_host'],self.config['broker_port']), json=task_completed)

        # Reset results counter for next task
        self.results_counter = 0