class JointInfo(object):
    """
    Definition of joint ordering
    """

    root = 0
    thumb_mcp = 1
    index_mcp = 2
    middle_mcp = 3
    ring_mcp = 4
    pinky_mcp = 5
    thumb_pip = 6
    index_pip = 7
    middle_pip = 8
    ring_pip = 9
    pinky_pip = 10
    thumb_dip = 11
    index_dip = 12
    middle_dip = 13
    ring_dip = 14
    pinky_dip = 15
    thumb_tip = 16
    index_tip = 17
    middle_tip = 18
    ring_tip = 19
    pinky_tip = 20

    root_mano = 0  
    thumb_mcp_mano = 17
    index_mcp_mano = 1  
    middle_mcp_mano = 5 
    ring_mcp_mano = 13
    pinky_mcp_mano = 9 
    thumb_pip_mano = 18
    index_pip_mano = 2 
    middle_pip_mano = 6 
    ring_pip_mano = 14
    pinky_pip_mano = 10 
    thumb_dip_mano = 19
    index_dip_mano = 3 
    middle_dip_mano = 7 
    ring_dip_mano = 15
    pinky_dip_mano = 11  
    thumb_tip_mano = 20
    index_tip_mano = 4 
    middle_tip_mano = 8 
    ring_tip_mano = 16
    pinky_tip_mano = 12

    
    idx_to_name = {}
    idx_to_name[root] = "root"
    idx_to_name[thumb_mcp] = "thumb_mcp"
    idx_to_name[index_mcp] = "index_finger_mcp"
    idx_to_name[middle_mcp] = "middle_finger_mcp"
    idx_to_name[ring_mcp] = "ring_finger_mcp"
    idx_to_name[pinky_mcp] = "pinky_mcp"
    idx_to_name[thumb_pip] = "thumb_pip"
    idx_to_name[index_pip] = "index_finger_pip"
    idx_to_name[middle_pip] = "middle_finger_pip"
    idx_to_name[ring_pip] = "ring_finger_pip"
    idx_to_name[pinky_pip] = "pinky_pip"
    idx_to_name[thumb_dip] = "thumb_dip"
    idx_to_name[index_dip] = "index_finger_dip"
    idx_to_name[middle_dip] = "middle_finger_dip"
    idx_to_name[ring_dip] = "ring_finger_dip"
    idx_to_name[pinky_dip] = "pinky_dip"
    idx_to_name[thumb_tip] = "thumb_tip"
    idx_to_name[index_tip] = "index_finger_tip"
    idx_to_name[middle_tip] = "middle_finger_tip"
    idx_to_name[ring_tip] = "ring_finger_tip"
    idx_to_name[pinky_tip] = "pinky_tip"

    def __getitem__(self, key):
        joint_idx = getattr(self, str(key))

        return joint_idx
