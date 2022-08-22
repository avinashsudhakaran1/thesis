import labelbox
# Enter your Labelbox API key here
LB_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbDZ4c25vYzIxbmwzMDc1aGJhbGVjb3huIiwib3JnYW5pemF0aW9uSWQiOiJjbDZ4c25vYm8xbmwyMDc1aDRubmJkMm4wIiwiYXBpS2V5SWQiOiJjbDc0b2VxY3U0dnZsMDcwa2dpbTBhZTN6Iiwic2VjcmV0IjoiYjExZjE1NTcyNzNlMTMwYjQzYzQ3ZmFhZTFlZWY5MGEiLCJpYXQiOjE2NjExNjc3ODUsImV4cCI6MjI5MjMxOTc4NX0.lCMCqLCf3EEgtnVdB2D9PlKwdPs2f-aseg99bsCvhdM"
# Create Labelbox client
lb = labelbox.Client(api_key=LB_API_KEY)
# Get project by ID
project = lb.get_project('cl72s9m8xbtlb0738azvvgy2b')
# Export image and text data as an annotation generator:
labels = project.label_generator()

