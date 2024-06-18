




def map(s, max_gesture_probs, target_objects, self.ap)
        
    if len(target_objects) == 0: # no object added, no pointing was made
        ''' Object not given -> use eef position '''
        focus_point = s.r.eef_position_real
    else:
        object_name_1 = target_objects[0]
        if s.get_object_by_name(object_name_1) is None:
            print(f"Object target is not in the scene!, object name: {object_name_1} objects: {s.O}")
            GestureSentence.clearing()
            return None
        focus_point = s.get_object_by_name(object_name_1).position_real

    if focus_point is None: return

    # target object to focus point {target_objects} -> {focus_point}

    sr = s.to_ros(SceneRos())
    sr.focus_point = np.array(focus_point, dtype=float)
    print(f"[INFO] Aggregated gestures: {list(gestures_queue_proc)}")

    time.sleep(0.01)

    g2i_tester = G2IRosNode(init_node=False, inference_type='1to1', load_model='M3v10_D6.pkl', ignore_unfeasible=True)

    gestures = gl.gd.gestures_queue_to_ros(gestures_queue_proc, GesturesRos())
    response = g2i_tester.G2I_service_callback( \
        G2I.Request(gestures=gestures, scene=sr),
        G2I.Response()
        )

    response_intent = response.intent




    self.gesture_sentence_publisher_mapped.publish(GestureSentence.export_mapped_to_HRICommand(s, ret))













