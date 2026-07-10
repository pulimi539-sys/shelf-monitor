from roboflow import Roboflow
rf = Roboflow(api_key="2lsB6R8zUuyUbbOMYCSy")
project = rf.workspace("college-project-gazum").project("shelf-retail-prototype")
version = project.version(2)
dataset = version.download("yolov8")
                