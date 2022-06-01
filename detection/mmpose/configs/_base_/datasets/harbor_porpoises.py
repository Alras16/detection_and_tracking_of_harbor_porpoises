dataset_info = dict(
    dataset_name='harbor_porpoises',
        paper_info=dict(
            author='Alexander Rasmussen',
            title='Detection and Tracking of Harbour Porpoises in UAV Recordings',
            year='2022',
        ),
    keypoint_info={
        0:
        dict(name='fin', id=0, color=[191, 92, 77], type='upper', swap=''), 
        1:
        dict(name='head', id=1, color=[217, 145, 0], type='lower', swap=''),
        2:
        dict(name='tail', id=2, color=[77, 128, 104], type='lower', swap='')
    },
    skeleton_info={
        0:
        dict(link=('head', 'fin'), id=0, color=[173, 127, 168]),
        1:
        dict(link=('fin', 'tail'), id=0, color=[173, 127, 168])
    },
    joint_weights=[
        1., 1., 1.
    ],
    sigmas=[
        0.025, 0.062, 0.079
    ])

