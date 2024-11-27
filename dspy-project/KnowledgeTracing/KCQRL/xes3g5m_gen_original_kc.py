import json
import requests
import hashlib
import random
import time


def baidu_translate(text, from_lang="en", to_lang="zh", app_id="your_app_id", secret_key="your_secret_key"):
    # API 请求地址
    url = "https://fanyi-api.baidu.com/api/trans/vip/translate"

    # 生成随机数用于签名
    salt = str(random.randint(32768, 65536))

    # 构造签名：md5(appid + text + salt + 密钥)
    sign = hashlib.md5((app_id + text + salt + secret_key).encode("utf-8")).hexdigest()

    # 构造请求参数
    params = {
        "q": text,
        "from": from_lang,
        "to": to_lang,
        "appid": app_id,
        "salt": salt,
        "sign": sign
    }

    # 发送请求
    response = requests.get(url, params=params)
    result = response.json()

    # 检查响应结果
    if "trans_result" in result:
        # 提取并返回翻译结果
        translated_text = result["trans_result"][0]["dst"]
        return translated_text
    else:
        # 处理错误
        error_msg = result.get("error_msg", "Error occurred")
        print("Translation failed:", error_msg)
        return None


if __name__ == "__main__":
    kc_meta_path = "/Users/dream/myProjects/dlkt/lab/dataset_raw/xes3g5m/metadata/kc_routes_map.json"
    json_file_dataset_path = "/Users/dream/projects/KCQRL-main/data/XES3G5M/metadata/questions_translated_kc_sol_annotated_mapped.json"
    questions_translated_path = "/Users/dream/projects/KCQRL-main/data/XES3G5M/metadata/questions_translated_kc_sol_annotated_mapped_original_kc.json"
    kc_clusters_path = "/Users/dream/projects/KCQRL-main/data/XES3G5M/metadata/kc_no_clusters_original_kc.json"

    with open(kc_meta_path, 'r') as f:
        kc_data = json.load(f)

    with open(json_file_dataset_path, 'r') as f:
        data = json.load(f)

    kc_translation_dict = {}
    kc_clusters_hdbscan = {}
    for c_id, kc in kc_data.items():
        if c_id == "865":
            break
        time.sleep(1)
        kc = kc.strip()
        kc_translated = baidu_translate(kc, from_lang="zh", to_lang="en", app_id="20241030002189241",
                                        secret_key="di4U2m3rvH2lXVZ2ymB6")
        kc_translation_dict[kc] = kc_translated
        kc_clusters_hdbscan[c_id] = [kc_translated]

    for q_id, q in data.items():
        data[q_id]["mapping_step_kc_gpt-4o"] = ", ".join([f"{step + 1}-1" for step in range(len(q["step_by_step_solution_list"]))])
        kcs = q['kc_routes']
        kcs_translated = []
        for kc in kcs:
            kc = kc.split("----")[-1].strip()
            kcs_translated.append(kc_translation_dict[kc])
        data[q_id]["knowledge_concepts_list"] = kcs_translated
        data[q_id]["knowledge_concepts_text"] = "\n".join(kcs_translated)

    with open(questions_translated_path, 'w') as f:
        json.dump(data, f)

    with open(kc_clusters_path, 'w') as f:
        json.dump(kc_clusters_hdbscan, f)
