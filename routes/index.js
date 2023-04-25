var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index');
});


router.get('/chat',function(req,res,next){
  res.render('chat');
});

router.get('/quote',function(req,res,next){
  res.render('quote');
});

router.get('/about',function(req,res,next){
  res.render('about');
});

router.get('/jokes',function(req,res,next){
  res.render('jokes');
});

module.exports = router;
