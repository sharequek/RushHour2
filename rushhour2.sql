select * from listings order by url desc;
select * from listings where id = 323;
select count(*) from listings;
select * from listing_photos;
select count(*) from listing_photos where type = 'floor_plan';
select * from listing_features;

select l.id, l.url, l.ocr_sqft_extracted, l.ocr_sqft_confidence, l.ocr_sqft_source_text, l.ocr_sqft_confidence, l.ocr_sqft_engine, lp.url
from listings l
join listing_photos lp on l.id = lp.listing_id and l.ocr_sqft_source_photo_id = lp.id
where l.ocr_sqft_extracted is not null
order by l.ocr_sqft_confidence;

select * from listings where ocr_sqft_extracted is not null order by id;

select l.id, l.url, l.ocr_sqft_extracted, l.ocr_sqft_confidence, l.ocr_sqft_source_text, l.ocr_sqft_confidence, l.ocr_sqft_engine, lp.url
from listings l
join listing_photos lp on l.id = lp.listing_id
where l.ocr_sqft_extracted is null and lp.type = 'floor_plan'
order by l.id;

select * from listings where combined_address like '%89 Hicks%';
select * from listing_photos where listing_id = 415;
select * from listing_photos where url like '%https://photos.zillowstatic.com/fp/44973d41b304789f1984a7fb87f95dde%';