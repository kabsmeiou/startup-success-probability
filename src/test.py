import pickle
import numpy as np

model_file=f'startup-success-predictor.bin'

with open(model_file, 'rb') as f_in:
    model, dv = pickle.load(f_in)

test_data = {'mass_funding_type': 'prize',
 'project_category': 'other',
 'funding_method': 'all-or-nothing',
 'project_supported': 1,
 'number_of_projects_owned': 1,
 'number_of_teams': 0,
 'project_duration': 62,
 'promo_video': 1,
 'promo_video_length': 104,
 'image_count': 1,
 'faq': 0,
 'updates': 4,
 'comments': 0,
 'reward_count': 11,
 'project_member_count': 4,
 'website': 0,
 'social_media': 1,
 'social_media_count': 3,
 'social_media_followers': 274,
 'total_tags': 0,
 'target_amount': 40000,
 'backer_count': 150
}
test_data['log_backer_count'] = np.log1p(test_data['backer_count'])
del test_data['backer_count']
X = dv.transform(test_data)
probability_of_success = model.predict_proba(X)[:,1].round(4)[0] * 100
print(f"Success probability of the project: {probability_of_success:.2f}")
