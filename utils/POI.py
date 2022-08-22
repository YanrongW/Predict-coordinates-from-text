import re


class Poi(object):
    def __init__(self, **kwargs):
        self.id = None
        self.name = None
        self._adr = None
        self.lat = None  # 纬度 geo的第二项
        self.lng = None  # 经度 geo的第一项
        self.poi_address_parts = None
        self.name_components_result = None
        self.s2_coder = None
        self.no_conflict = -1

    @property
    def adr(self):
        return self._adr

    @adr.setter
    def adr(self, value):
        citys = ["泉州" "常州" "温州" "无锡" "广州" "大连"
                 "佛山" "武汉" ]
        provinces = ["福建" "江苏" "浙江" "江苏" "广东" "辽宁"
                     "广东" "湖北"]
        delete_province = False
        for i, city in enumerate(citys):
            if city in value and value[:len(provinces[i])] == provinces[i]:
                delete_province = True
                break
        if delete_province:
            pattern = ".{2,3}省"
            match = re.match(pattern, value)
            if (match is not None):
                start, end = match.span()
                if(start == 0 and end - start <=4):
                    value = value[end:]
        self._adr = value
