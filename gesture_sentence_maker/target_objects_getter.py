




def get_target_object(hand='lr', mode='original'):
    
    s = get_scene()
    
    def most_frequent(List):
        return max(set(List), key = List.count)

    _, nameobj_, object_distances = RealRobotConvenience.deictic(hand, s)
    if nameobj_ is None:
        print("No objects on scene!")
        return 'q', None

    RealRobotConvenience.go_on_top_of_object_modular(nameobj_, s)
    else: raise Exception()
    
    return nameobj_, object_distances
