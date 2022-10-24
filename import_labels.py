import labelbox
# Enter your Labelbox API key here
LB_API_KEY = ""
# Create Labelbox client
lb = labelbox.Client(api_key=LB_API_KEY)
# Get project by ID
project = lb.get_project('')
# Export image and text data as an annotation generator:
labels = project.label_generator()

