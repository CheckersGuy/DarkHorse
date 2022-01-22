//
// Created by robin on 20.01.22.
//

#ifndef READING_LRUCACHE_H
#define READING_LRUCACHE_H

#include <list>
#include <unordered_map>

template<class K, class T, typename Hasher = std::hash<K>>
class Cache {
private:

    mutable std::list<std::pair<K, T>> liste;
    std::unordered_map<K, decltype(liste.begin()), Hasher> map;
    size_t cacheSize;
public:

    explicit Cache(size_t size) : cacheSize(size) {}

    void put(K key, T value) {
        auto iter = map.find(key);

        if (iter == map.end()) {
            //not in the list
            if (map.size() > cacheSize) {
                auto &back = liste.back();
                map.erase(back.first);
                liste.pop_back();
            }
            liste.push_front(std::make_pair(key, value));
            map.insert(std::make_pair(key, liste.begin()));
        } else {
            liste.splice(liste.begin(), liste, iter->second);
        }

    }

    std::optional<T> get(K key) const {
        auto iter = map.find(key);
        if (iter != map.end()) {
            //need to move it to the front though
            liste.splice(liste.begin(), liste, iter->second);
            return std::make_optional(iter->second->second);
        }
        return std::nullopt;
    }

    bool has(K key) {
        auto iter = map.find(key);
        return iter != map.end();
    }

    auto begin() {
        return liste.begin();
    }

    auto end() {
        return liste.end();
    }
};

#endif //READING_LRUCACHE_H
