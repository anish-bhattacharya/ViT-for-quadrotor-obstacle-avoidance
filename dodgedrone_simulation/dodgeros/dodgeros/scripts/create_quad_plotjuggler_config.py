#!/usr/bin/env python3
import rospy
from string import Template


if __name__ == "__main__":
    rospy.init_node('plotjugglerconfig', anonymous=True)
    quad_name = rospy.get_param('~quad_name')
    template_file = rospy.get_param('~template_file')
    output_config_file = rospy.get_param('~output_config_file')

    rospy.loginfo("create config for %s from template %s"%(quad_name,template_file))

    subs_map = {"quad_name":quad_name}
    with open(template_file, 'r') as f:
        src = Template(f.read())
        result = src.substitute(subs_map)
        
        with open(output_config_file,'w') as fout:
            fout.write(result)
            rospy.loginfo("result wrote to %s"%(output_config_file))
