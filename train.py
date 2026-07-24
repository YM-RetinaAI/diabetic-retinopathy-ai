def main_manager():
    set_seed(42)
    
    train_patients, test_patients = split_patients_fairly(all_data)
    
    train_nurse = DataLoader(train_patients, batch_size=16)
    val_nurse = DataLoader(test_patients, batch_size=16)
    
    doctor = build_model(pretrained=True)
    
    optimizer = setup_learning_speeds(doctor.backbone_speed, doctor.head_speed)
    
    best_exam_score = -1.0
    
    for day in range(1, 30):
        train_one_epoch(doctor, train_nurse, optimizer)
        
        exam_score = evaluate(doctor, val_nurse)
        
        if exam_score > best_exam_score:
            best_exam_score = exam_score
            save_brain_to_hard_drive(doctor)
            print("رکورد جدید ثبت شد!")