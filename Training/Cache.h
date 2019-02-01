//
// Created by robin on 1/17/19.
//

#ifndef TRAINING_CACHE_H
#define TRAINING_CACHE_H



#include <list>
#include <unordered_map>

enum class CacheMode{LRU,MRU};

template<CacheMode mode ,class K, class T>
class Cache {

private:

    std::list<std::pair<K, T>> liste;

    std::unordered_map<K, decltype(liste.begin())> map;
    size_t cacheSize;
public:

    Cache(size_t size) : cacheSize(size) {}

    void put(K key, T value) {
        auto iter = map.find(key);

        if (iter == map.end()) {
            //not in the list
            if(map.size()>cacheSize){
                auto back =liste.end();
                liste.pop_back();
                map.erase(back->first);
            }
            liste.push_front(std::make_pair(key,value));
            map.insert(std::make_pair(key,liste.begin()));
        }else{
            liste.splice(liste.begin(), liste, iter->second);
        }

    }

    std::optional<T> get(K key){
        std::optional<T> result;
        auto iter = map.find(key);
        if(iter!=map.end()){
            //need to move it to the front though
            liste.splice(liste.begin(),liste,iter->second);
            result =iter->second->second;
        }
        return result;
    }

    auto begin(){
        return liste.begin();
    }

    auto end(){
        return liste.end();
    }
};


#endif //TRAINING_CACHE_H
