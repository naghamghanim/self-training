





sports_vocab = {'تمرين', 'لياقة', 'مباراة', 'فريق'} 
health_vocab = {'تمرين', 'لياقة', 'علاج', 'تشخيص'}
overlap = sports_vocab.intersection(health_vocab)  # {'تمرين', 'لياقة'}
print(overlap)