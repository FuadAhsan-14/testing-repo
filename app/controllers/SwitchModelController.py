import os
import sys
# from app.controllers.FakturControllerAX import fakturax_controller
from app.controllers.ObjectaxController import objectax_controller
# from app.controllers.FakturControllerAXBackup import fakturax_controller_backup
from app.controllers.ObjectaxBackupController import objectaxBackup_controller
from app.utils.Http.HttpResponseUtils import response_success, response_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config.ratelimit import redis_connection

class choosemodel:
    def __init__(self):
        pass
        
    async def object_model(self, files, po_number):
        try:
            model = await redis_connection.get("object_model")
            if model == "primary":
                object_switch_controller = await objectax_controller.run_workflow(files, po_number)
                # object_switch_controller = await object_warehouse_controller_ax.anomaly_graph(files, po_number)
                return object_switch_controller
            elif model =="backup":
                object_switch_controller = await objectaxBackup_controller.run_workflow(files, po_number)
                return object_switch_controller
        except Exception as e:
            raise response_error(str(e))

switch_model_controller = choosemodel()

