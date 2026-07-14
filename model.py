class SimpleDRModel:
    def __init__(self):
        
       
        self.vision_extractor = load_timm_model(name="efficientnet", pretrained=True)
        
        num_features = self.vision_extractor.count_features()

      
       
        self.specialist_brain = Chain_of_Layers(
            Linear(input=num_features, output=512),
            
            Forget_Random_Neurons(rate=40%),
            
            Linear(input=512, output=5_Classes)
        )

  
    def forward(self, image):
        raw_features = self.vision_extractor(image)
        
        final_diagnosis = self.specialist_brain(raw_features)
        
        return final_diagnosis


def build_and_setup_doctor():
    doctor = SimpleDRModel()
    return doctor.send_to_GPU()