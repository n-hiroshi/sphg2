# -*- coding: utf-8 -*-
"""
Utility function to delete mutiplicated elements of the list
"""

class unique: 
    @classmethod
    def unique(self,list_in):
        list_out=[]
        for elem in list_in:
            if not(elem in list_out):
                list_out.append(elem)
        return list_out



    
  
